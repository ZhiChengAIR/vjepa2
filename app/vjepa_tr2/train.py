# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import time
import tqdm

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.vjepa_tr2.tr2 import init_data
from app.vjepa_tr2.transforms import make_transforms
from app.vjepa_tr2.utils import init_opt, init_video_model, load_checkpoint, load_pretrained
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from src.utils.swanlab_keeper import SwanlabKeeper
from src.utils.world_model_wrapper import WorldModel
from src.utils.metrics import plot_trajectory_comparison

# --
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    # -- META: 元数据配置
    folder = args.get("folder")  # 保存模型的文件夹路径
    cfgs_meta = args.get("meta")  # 获取元配置
    r_file = cfgs_meta.get("resume_checkpoint", None)  # 恢复训练检查点文件
    p_file = cfgs_meta.get("pretrain_checkpoint", None)  # 预训练检查点文件
    load_predictor = cfgs_meta.get("load_predictor", False)  # 是否加载预测器
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")  # 上下文编码器键名
    target_encoder_key = cfgs_meta.get("target_encoder_key", "target_encoder")  # 目标编码器键名
    load_encoder = cfgs_meta.get("load_encoder", True)  # 是否加载编码器
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)  # 随机种子
    save_every_freq = cfgs_meta.get("save_every_freq", -1)  # 保存频率
    skip_batches = cfgs_meta.get("skip_batches", -1)  # 跳过的批次数量
    use_sdpa = cfgs_meta.get("use_sdpa", False)  # 是否使用SDPA
    sync_gc = cfgs_meta.get("sync_gc", False)  # 是否同步垃圾回收
    which_dtype = cfgs_meta.get("dtype")  # 数据类型
    val_every_freq = cfgs_meta.get("val_every_freq", 1)  # 验证频率
    compute_action_mse_every = cfgs_meta.get("compute_action_mse_every", 1)  # 计算动作均方误差的频率
    logger.info(f"{which_dtype=}")

    # 根据配置设置数据类型和混合精度
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MODEL: 模型配置
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)  # 是否编译模型
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)  # 是否使用激活检查点
    model_name = cfgs_model.get("model_name")  # 模型名称
    pred_depth = cfgs_model.get("pred_depth")  # 预测器深度
    pred_num_heads = cfgs_model.get("pred_num_heads", None)  # 预测器头数
    pred_embed_dim = cfgs_model.get("pred_embed_dim")  # 预测器嵌入维度
    pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)  # 预测器是否帧因果
    uniform_power = cfgs_model.get("uniform_power", False)  # 是否均匀功率
    use_rope = cfgs_model.get("use_rope", False)  # 是否使用RoPE
    use_silu = cfgs_model.get("use_silu", False)  # 是否使用SiLU激活
    use_pred_silu = cfgs_model.get("use_pred_silu", False)  # 预测器是否使用SiLU
    wide_silu = cfgs_model.get("wide_silu", True)  # 是否使用宽SiLU
    use_extrinsics = cfgs_model.get("use_extrinsics", False)  # 是否使用外部参数
  # -- DATA: 数据配置
    cfgs_data = args.get("data")
    datasets = cfgs_data.get("datasets", [])  # 数据集列表
    dataset_path = datasets[0]  # 数据集路径
    dataset_fpcs = cfgs_data.get("dataset_fpcs")  # 每段视频的帧数
    max_num_frames = max(dataset_fpcs)  # 最大帧数
    camera_frame = cfgs_data.get("camera_frame", False)  # 是否使用相机坐标系
    camera_views = cfgs_data.get("camera_views", ["left_mp4_path"])  # 相机视角
    stereo_view = cfgs_data.get("stereo_view", False)  # 是否立体视图
    batch_size = cfgs_data.get("batch_size")  # 批次大小
    tubelet_size = cfgs_data.get("tubelet_size")  # 管状体大小
    fps = cfgs_data.get("fps")  # 帧率
    crop_size = cfgs_data.get("crop_size", 256)  # 裁剪尺寸
    patch_size = cfgs_data.get("patch_size")  # 补丁大小
    pin_mem = cfgs_data.get("pin_mem", False)  # 是否固定内存
    num_workers = cfgs_data.get("num_workers", 1)  # 数据加载工作线程数
    persistent_workers = cfgs_data.get("persistent_workers", True)  # 是否保持工作线程
    val_ratio = cfgs_data.get("val_ratio", 0.05)  # 验证集比例

    # -- DATA AUGS: 数据增强配置
    cfgs_data_aug = args.get("data_aug")
    horizontal_flip = cfgs_data_aug.get("horizontal_flip", False)  # 水平翻转
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])  # 随机调整宽高比范围
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])  # 随机调整比例范围
    motion_shift = cfgs_data_aug.get("motion_shift", False)  # 运动偏移
    reprob = cfgs_data_aug.get("reprob", 0.0)  # 随机擦除概率
    use_aa = cfgs_data_aug.get("auto_augment", False)  # 自动增强

    # -- LOSS: 损失函数配置
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")  # 损失指数
    normalize_reps = cfgs_loss.get("normalize_reps")  # 是否归一化表示
    auto_steps = min(cfgs_loss.get("auto_steps", 1), max_num_frames)  # 自动步数
    # --
    tokens_per_frame = int((crop_size // patch_size) ** 2)  # 每帧的token数量

    # -- OPTIMIZATION: 优化配置
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)  # 每epoch迭代次数
    wd = float(cfgs_opt.get("weight_decay"))  # 权重衰减
    final_wd = float(cfgs_opt.get("final_weight_decay"))  # 最终权重衰减
    num_epochs = cfgs_opt.get("epochs")  # 总epoch数
    anneal = cfgs_opt.get("anneal")  # 退火策略
    warmup = cfgs_opt.get("warmup")  # 预热步数
    start_lr = cfgs_opt.get("start_lr")  # 初始学习率
    lr = cfgs_opt.get("lr")  # 学习率
    final_lr = cfgs_opt.get("final_lr")  # 最终学习率
    enc_lr_scale = cfgs_opt.get("enc_lr_scale", 1.0)  # 编码器学习率缩放
    betas = cfgs_opt.get("betas", (0.9, 0.999))  # Adam优化器参数
    eps = cfgs_opt.get("eps", 1.0e-8)  # Adam优化器参数
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    resume_path = os.path.join(folder, r_file) if r_file is not None else latest_path
    if not os.path.exists(resume_path):
        resume_path = None

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%d", "step"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
        mode="+a",
    )

    if rank == 0 :
        swanlab_runner = SwanlabKeeper(
            config=args
        )

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        device=device,
        patch_size=patch_size,
        max_num_frames=512,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        action_embed_dim=20,
        state_embed_dim=20,
        pred_is_frame_causal=pred_is_frame_causal,
        use_extrinsics=use_extrinsics,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    target_encoder = copy.deepcopy(encoder)

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()

    video_collator = torch.utils.data.default_collate
    transform = make_transforms(
        random_horizontal_flip=horizontal_flip,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers
    (train_loader, val_loader, train_sampler, val_sampler, dataset) = init_data(
        data_path=dataset_path,
        batch_size=batch_size,
        frames_per_clip=max_num_frames,
        tubelet_size=1,
        fps=fps,
        camera_views=camera_views,
        camera_frame=camera_frame,
        stereo_view=stereo_view,
        transform=transform,
        collator=video_collator,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        rank=rank,
        val_ratio=val_ratio,
    )
    _dlen = len(train_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataest length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        enc_lr_scale=enc_lr_scale,
        iterations_per_epoch=ipe,
        anneal=anneal,
        warmup=warmup,
        num_epochs=num_epochs,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- looad pretrained weights
    encoder, predictor, target_encoder = load_pretrained(
        r_path=p_file,
        encoder=encoder,
        predictor=predictor,
        context_encoder_key=context_encoder_key,
        target_encoder_key=target_encoder_key,
        target_encoder=target_encoder,
        load_predictor=load_predictor,
        load_encoder=load_encoder,
    )

    ac_world_model = WorldModel(
                encoder=target_encoder,
                predictor=predictor,
                tokens_per_frame=tokens_per_frame,
                rotation_transformer=dataset.rotation_transformer,  
                mpc_args={
                    "rollout": 23,
                    "samples": 10,
                    "topk": 10,
                    "cem_steps": 5,
                    "momentum_mean_pose": 6,
                    "momentum_std_pose": 1.4,
                    "momentum_mean_rot": 0.2,
                    "momentum_std_rot": 0.05,
                    "momentum_mean_gripper": 0.0002,
                    "momentum_std_gripper": 0.0001,
                    "max_pose": 30,
                    "max_rot": 1,
                    "max_gripper": 0.007,
                },
                normalize_reps=True,
                device=device,
                app=args["app"]
            )

    start_epoch = 0
    # -- load training checkpoint
    if os.path.exists(latest_path):
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=resume_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    train_sampler.set_epoch(start_epoch)
    loader = iter(train_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        # -- update distributed-data-loader epoch

        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(train_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()



    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        jloss_meter = AverageMeter()
        sloss_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        # save batch for sampling
        train_sampling_batch = None
        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    train_sampling_batch = sample
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    train_sampler.set_epoch(epoch)
                    loader = iter(train_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                        raise e

            def load_clips(sample):
                clips = sample[0].to(device, non_blocking=True)  # [B C T H W]
                actions = sample[1].to(device, dtype=torch.float, non_blocking=True)  # [B T-1 7]
                states = sample[2].to(device, dtype=torch.float, non_blocking=True)  # [B T 7]
                extrinsics = None
                return (clips, actions, states, extrinsics)

            clips, actions, states, extrinsics = load_clips(sample)
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def forward_target(c):
                with torch.no_grad():
                    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
                    h = target_encoder(c)
                    h = h.view(batch_size, max_num_frames, -1, h.size(-1)).flatten(1, 2)
                    if normalize_reps:
                        h = F.layer_norm(h, (h.size(-1),))
                    return h

            def forward_predictions(z):

                def _step_predictor(_z, _a, _s, _e):
                    _z = predictor(_z, _a, _s, _e)
                    if normalize_reps:
                        _z = F.layer_norm(_z, (_z.size(-1),))
                    return _z

                # -- one step of predictor with teacher forcing
                _z, _a, _s, _e = z[:, :-tokens_per_frame], actions, states[:, :-1], None
                z_tf = _step_predictor(_z, _a, _s, _e)

                # -- full auto-regressive rollouts of predictor
                _z = torch.cat([z[:, :tokens_per_frame], z_tf[:, tokens_per_frame : 2 * tokens_per_frame]], dim=1)
                for n in range(1, auto_steps):
                    _a, _s, _e = actions[:, : n + 1], states[:, : n + 1], None
                    _z_nxt = _step_predictor(_z, _a, _s, _e)[:, -tokens_per_frame:]
                    _z = torch.cat([_z, _z_nxt], dim=1)
                z_ar = _z[:, tokens_per_frame:]

                return z_tf, z_ar

            def loss_fn(z, h):
                _h = h[:, tokens_per_frame : z.size(1) + tokens_per_frame]
                return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)
                    z_tf, z_ar = forward_predictions(h)
                    jloss = loss_fn(z_tf, h)
                    sloss = loss_fn(z_ar, h)
                    loss = jloss + sloss

                # Step 2. Backward & step
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                return (
                    float(loss),
                    float(jloss),
                    float(sloss),
                    _new_lr,
                    _new_wd,
                )

            (
                loss,
                jloss,
                sloss,
                _new_lr,
                _new_wd,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            jloss_meter.update(jloss)
            sloss_meter.update(sloss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)


            # -- Logging
            def log_stats():
                csv_logger.log(epoch, itr, loss, iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms)
                global_step = epoch * ipe + itr
                step_log = {
                    "epoch": epoch + itr/ipe,
                    "itr": itr,
                    "loss": loss,
                    "iter_elapsed_time_ms": iter_elapsed_time_ms,
                    "gpu_etime_ms": gpu_etime_ms,
                    "data_elapsed_time_ms": data_elapsed_time_ms,
                }
                swanlab_runner.log(global_step=global_step, step_log=step_log)
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f [%.2f, %.2f] "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            jloss_meter.avg,
                            sloss_meter.avg,
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        # run validation
        if (epoch % val_every_freq) == 0: 
            with torch.no_grad():
                val_losses = list()
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):

                    with tqdm.tqdm(val_loader, desc=f"Validation epoch {epoch}", 
                                leave=False, mininterval=1) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            clips, actions, states, extrinsics = load_clips(batch)
                            h = forward_target(clips)
                            z_tf, z_ar = forward_predictions(h)
                            jloss = loss_fn(z_tf, h)
                            sloss = loss_fn(z_ar, h)
                            loss = jloss + sloss
                            val_losses.append(float(loss))
            step_log = {}
            if  len(val_losses) > 0:
                val_loss = torch.mean(torch.tensor(val_losses)).item()
                # log epoch average validation loss
                step_log['val_loss'] = val_loss
                global_step = (epoch + 1) * ipe
                swanlab_runner.log(global_step=global_step, step_log=step_log)

        if (epoch % compute_action_mse_every) == 0:
            with torch.no_grad():
                # -- compute action mse
                clips, actions, states, extrinsics = load_clips(train_sampling_batch)
                actions = actions.cpu().numpy().squeeze(0)
                h = ac_world_model.encode(clips)
                idx = 0
                pred_actions = ac_world_model.infer_next_action(h[:,idx*tokens_per_frame:(idx+1)*tokens_per_frame], states[:, idx], h[:,(idx+1)*tokens_per_frame:(idx+2)*tokens_per_frame]).cpu().numpy()
                mse = np.mean((actions - pred_actions) ** 2)
                
                # plot left
                title = f"Epoch{epoch}_left-action_mse:{mse:.4f}"
                fig = plot_trajectory_comparison(
                    gt_xyz=actions[:, :3],
                    pred_xyz=pred_actions[:, :3],
                    title=title,
                )
                filename = f"{title}.png"
                output_folder = os.path.join(folder, "plots")
                os.makedirs(output_folder, exist_ok=True)
                fig.savefig(os.path.join(output_folder, filename))

                # plot right
                title = f"Epoch{epoch}_right-action_mse:{mse:.4f}"
                fig = plot_trajectory_comparison(
                    gt_xyz=actions[:, 10:13],
                    pred_xyz=pred_actions[:, 10:13],
                    title=title,
                )
                filename = f"{title}.png"
                fig.savefig(os.path.join(output_folder, filename))
                global_step = (epoch + 1) * ipe
                swanlab_runner.log(global_step=global_step, step_log={"actions_mse": mse})
                logger.info("avg. action mse %.3f" % mse)

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        # -- Save Last
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)
