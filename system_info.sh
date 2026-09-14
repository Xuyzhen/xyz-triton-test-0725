#!/bin/bash
# =============================================================================
# system_info.sh - Linux 服务器系统配置与现状采集脚本
# 覆盖：基本信息 / CPU / 内存 / NPU(昇腾) / GPU(NVIDIA) / 存储 / 网络 / 系统配置
# 用法：bash system_info.sh    （无需 root 也能运行，root 下信息更全）
# =============================================================================

set -u

# ---------- 颜色 ----------
C_TITLE='\033[1;34m'
C_SUB='\033[1;32m'
C_WARN='\033[1;33m'
C_RESET='\033[0m'

section() {
    echo ""
    echo -e "${C_TITLE}============================================================${C_RESET}"
    echo -e "${C_TITLE}== $*${C_RESET}"
    echo -e "${C_TITLE}============================================================${C_RESET}"
}

sub() {
    echo -e "${C_SUB}---- $* ----${C_RESET}"
}

have() { command -v "$1" >/dev/null 2>&1; }

run() {
    local desc="$1"; shift
    echo -e "${C_WARN}[$desc]${C_RESET}"
    "$@" 2>/dev/null || echo "  (无法执行 $desc)"
}

# =============================================================================
section "1. 基本信息"
# =============================================================================
sub "主机名 / 系统 / 内核"
run "主机名" hostname
run "系统版本" cat /etc/os-release
run "内核版本" uname -a
run "运行时间/负载" uptime
run "当前时间" date

sub "硬件平台"
run "架构" uname -m
run "厂商/产品" hostnamectl 2>/dev/null

# =============================================================================
section "2. CPU"
# =============================================================================
if have lscpu; then
    sub "CPU 总览 (lscpu)"
    lscpu
else
    sub "CPU 信息 (/proc/cpuinfo)"
    grep -E "model name|physical id|cpu cores|siblings" /proc/cpuinfo | sort -u
fi

sub "CPU 数量"
echo "  逻辑核(cpu): $(grep -c ^processor /proc/cpuinfo)"
echo "  物理socket:  $(grep 'physical id' /proc/cpuinfo | sort -u | wc -l)"

sub "NUMA 节点"
if have numactl; then
    numactl --hardware
else
    run "NUMA 节点" ls /sys/devices/system/node/ 2>/dev/null
fi

sub "CPU 繁忙/频率"
run "cpu占用" top -bn1 | head -n 20
if have cpupower; then run "频率" cpupower frequency-info; fi

# =============================================================================
section "3. 内存"
# =============================================================================
sub "内存使用 (free)"
free -h

sub "/proc/meminfo 关键项"
grep -E "MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree|HugePages_Total|HugePages_Free|Hugepagesize" /proc/meminfo

sub "物理内存条 (dmidecode, 需 root)"
if have dmidecode; then
    run "DIMM 信息" dmidecode -t 17
else
    echo "  (dmidecode 未安装，跳过 DIMM 详情)"
fi

sub "NUMA 内存分布"
if have numactl; then numactl --hardware; fi

# =============================================================================
section "4. NPU (Ascend 昇腾)"
# =============================================================================
if have npu-smi; then
    sub "昇腾 NPU 状态 (npu-smi info)"
    run "npu-smi info" npu-smi info
    sub "昇腾 NPU 详细"
    run "npu-smi info -t board" npu-smi info -t board
fi

sub "其他加速卡 (通过 lspci 探测)"
if have lspci; then
    lspci | grep -iE "ascend|accelerator|npu|processing accelerators" || echo "  (未发现明显加速卡)"
fi

# =============================================================================
section "5. GPU (NVIDIA)"
# =============================================================================
if have nvidia-smi; then
    sub "NVIDIA GPU 状态"
    run "nvidia-smi" nvidia-smi
    sub "NVIDIA GPU 详细"
    run "nvidia-smi -q" nvidia-smi -q | head -n 120
else
    echo "  (未检测到 nvidia-smi，无 NVIDIA GPU 或驱动未装)"
fi

# =============================================================================
section "6. 存储 / 磁盘"
# =============================================================================
sub "文件系统容量 (df -h)"
df -hT

sub "inode 使用 (df -i)"
df -i

sub "块设备 (lsblk)"
if have lsblk; then
    run "块设备树" lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL
else
    run "块设备" cat /proc/partitions
fi

sub "NVMe / 磁盘详情"
if have nvme; then
    run "NVMe 列表" nvme list
fi
if have smartctl; then
    for d in $(lsblk -dno NAME 2>/dev/null); do
        run "SMART /dev/$d" smartctl -H "/dev/$d"
    done
else
    echo "  (smartctl 未安装，跳过健康检查)"
fi

sub "存储 I/O 现状"
if have iostat; then
    run "iostat" iostat -x 1 1
else
    echo "  (iostat 未安装 (sysstat)，跳过)"
fi

sub "已挂载的 RAID/LVM"
run "LVM 概况" lvs
run "RAID (mdstat)" cat /proc/mdstat

# =============================================================================
section "7. 网络"
# =============================================================================
sub "网卡 / IP 地址"
if have ip; then
    run "IP 地址" ip -br addr
    run "路由表" ip route
else
    run "IP 地址(ifconfig)" ifconfig
    run "路由表(route)" route -n
fi

sub "网卡速率 / 链路状态"
if have ethtool; then
    for dev in $(ls /sys/class/net/ | grep -vE "^lo$"); do
        echo -e "${C_WARN}[网口 $dev]${C_RESET}"
        ethtool "$dev" 2>/dev/null | grep -E "Speed|Duplex|Link detected" || echo "  (无信息)"
    done
else
    echo "  (ethtool 未安装，跳过)"
fi

sub "监听端口 / 连接状态"
run "监听端口" ss -tulnp

# =============================================================================
section "8. 系统配置"
# =============================================================================
sub "资源限制 (ulimit -a)"
ulimit -a

sub "内核参数 (sysctl 关键项)"
for k in vm.swappiness vm.dirty_ratio vm.overcommit_memory vm.max_map_count kernel.pid_max net.core.somaxconn net.ipv4.ip_forward; do
    echo "  $k = $(sysctl -n $k 2>/dev/null)"
done

sub "Transparent Huge Pages"
cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo "  (无THP信息)"

sub "SELinux"
if have getenforce; then
    echo "  SELinux 状态: $(getenforce 2>/dev/null)"
else
    echo "  (SELinux 未启用或未安装)"
fi

sub "防火墙"
if have systemctl && systemctl is-active firewalld >/dev/null 2>&1; then
    echo "  firewalld: active"
else
    echo "  firewalld: inactive / not installed"
fi

sub "系统服务/开机项 (可选，较多)"
run "已启用服务" systemctl list-unit-files --state=enabled 2>/dev/null | head -n 40

sub "日志错误巡检"
echo "  (如需，可补充 journalctl -p err -b 等)"

# =============================================================================
section "9. 汇总快照"
# =============================================================================
echo "采集完成时间: $(date)"
echo "主机: $(hostname)"
echo "内核: $(uname -r)"
echo "CPU 逻辑核: $(grep -c ^processor /proc/cpuinfo)"
echo "内存总量: $(grep MemTotal /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')"
echo ""
echo "提示：需要 root 权限才能完整获取 DIMM/SMART/ethtool 等信息。"