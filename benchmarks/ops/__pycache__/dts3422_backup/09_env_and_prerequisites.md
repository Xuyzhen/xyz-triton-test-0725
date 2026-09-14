# 09 环境前置条件与容器配置

## 一、测试环境

### 1.1 宿主机
- IP: 80.5.9.136
- 硬件: A3 16卡 (Atlas 800)
- OS: Linux
- 用户: root（已协调，他人未知 vllm 进程可杀，不允许改密码）

### 1.2 Docker 容器

| 容器名 | 用途 | vllm 版本 | vllm-ascend 安装方式 |
|--------|------|-----------|---------------------|
| xyz_dts_3422_main | main 分支测试 (可 checkout) | 0.27.1 | editable (`pip install -e .`) |
| xyz_dts_3422_v023 | v0.23 基准测试 (固定) | 0.23.0 | site-packages (不可 checkout) |
| xyz_aisbench_dts3422 | ais_bench 压测 | — | ais_bench installed |

### 1.3 关键路径
```
/mnt/weight/Kimi-K2.6-w4a8         — 主模型权重
/mnt/weight/Kimi-K2.5-DFlash        — draft 模型权重
/mnt/share/x30084275/dts_3422/vllm-ascend  — vllm-ascend 代码仓库
/mnt/share/x30084275/dts_3422/benchmark    — ais_bench 代码
```

### 1.4 网络通信
- serve 容器监听 8000 端口
- aisbench 容器通过 localhost:8000 访问
- 需设置 `no_proxy='127.0.0.1,0.0.0.0,localhost,local,.local,*.huawei.com'`

## 二、SSH 访问

### 2.1 从 Windows 访问
```powershell
# 使用 askpass 方式免密登录
$env:SSH_ASKPASS="c:\Users\x30084275\Desktop\git\26august\dts-3422\aisbench_kit\askpass.bat"
$env:SSH_ASKPASS_REQUIRE="force"
ssh -o StrictHostKeyChecking=no root@80.5.9.136 "命令"
```

### 2.2 askpass.bat 模板
```bat
@echo off
echo <密码>
```

## 三、环境变量

### 3.1 serve 容器环境变量
```bash
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=800
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export DYNAMIC_EPLB=true
export VLLM_USE_MODELSCOPE=False
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

### 3.2 aisbench 容器环境变量
```bash
export no_proxy='127.0.0.1,0.0.0.0,localhost,local,.local,*.huawei.com'
export NO_PROXY=$no_proxy
```

## 四、进程清理

### 4.1 标准清理命令（每次切换前）
```bash
# 在 main 和 v023 容器中各执行一次
docker exec $CTN pkill -9 -f spawn_main 2>/dev/null
docker exec $CTN pkill -9 -f resource_tracker 2>/dev/null
docker exec $CTN pkill -9 -f WorkerProc 2>/dev/null
docker exec $CTN pkill -9 -i -f vllm 2>/dev/null
```

### 4.2 NPU 空闲检查
```bash
# 检查 NPU 是否空闲（usage > 5000 视为忙）
npu_busy() {
    npu-smi info | grep -oE '[0-9]+ */ *65536' | awk -F'/' '{u=$1+0; if(u>5000) c++} END{print c+0}'
}
# 等待 NPU 空闲（最多 7.5 分钟）
for i in $(seq 1 30); do
    b=$(npu_busy)
    [ "$b" -eq 0 ] && break
    sleep 15
done
```

## 五、vllm-ascend 兼容性边界

| commit | pos | 日期 | 说明 |
|--------|-----|------|------|
| 36fcb76edc | 0 | — | v0.23.0 基准 |
| 2080fffa6 | 956 | 09-01 | 好锚点（稳定 1444） |
| 222677fc7 | 1045 | 09-04 | 坏锚点（波动 22%） |
| fd815467c | 1095 | 09-07 | **vllm 0.27.1 兼容边界** |
| 748acedfe | 1070 | 09-04 | xround 终点 / ver B 组 |

**注意**: bisect 窗口 (956-1045) 全部在兼容边界 (1095) 之前，可安全 checkout。

## 六、在别的设备上恢复执行

### 6.1 前置条件
1. 能 SSH 到 80.5.9.136（或有等效的 A3 机器）
2. 三个容器已配置好（或用 `06_executable_scripts/deploy_ver.sh` 重建）
3. 模型权重已挂载到 `/mnt/weight/`
4. vllm-ascend 代码仓库已 clone 到 `/mnt/share/x30084275/dts_3422/vllm-ascend`

### 6.2 恢复步骤
```bash
# 1. 部署脚本到远端
scp -r 06_executable_scripts/* root@80.5.9.136:/tmp/

# 2. 拷贝到容器
docker cp /tmp/serve_main_allon.sh xyz_dts_3422_main:/tmp/
docker cp /tmp/serve_v023_allon.sh xyz_dts_3422_v023:/tmp/
# ... 其他脚本

# 3. 执行 bisect
ssh root@80.5.9.136 "nohup bash /tmp/bisect2_runner.sh > /tmp/bisect2.log 2>&1 &"

# 4. 轮询进度
ssh root@80.5.9.136 "cat /tmp/bisect2_results.txt"
```
