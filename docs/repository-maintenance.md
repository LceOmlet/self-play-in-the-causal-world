# 代码与证据保存

继续维护用户指定的原仓库：
`https://github.com/LceOmlet/self-play-in-the-causal-world`。
现有历史保留，当前工作分支为 `codex/rl-correctness-official-dapo-20260907`。

服务器实际仓库位于
`/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831`；
本地完整 Git 副本位于
`C:\Users\Chen\Documents\ChatGPT\credit\self-play-in-the-causal-world`。
此前 `audit/rl_algorithm_fix_20260906/project` 只是文件镜像；后续本地代码维护应使用
完整 Git 副本，防止多份未提交镜像分叉。

后续每个有明确验收结论的改动应及时提交，并推送到同一工作分支，再用远端 ref
核对提交号。保留失败证据和未解决项，避免将通过范围扩大。合并主分支另按实际
验收与用户任务执行，不强推、不改写已有历史。大型权重和原始张量由审计目录保存，
代码、复现程序、轻量结果与来源哈希由 Git 保存。

本轮 ET-V2 内核改动及其一般密度推导、成对测量、最终回归和中间失败记录归档在
`research/et_v2_projection_20260907`。从本地已验收提交通过 Git bundle 向服务器
传递历史，再快进同一工作分支并推送 GitHub；不以覆盖文件代替分支和提交管理。
该目录的 `SHA256SUMS` 可用于检查归档字节完整性。

补丁含上游原始行尾，Git 已关闭补丁与研究归档的行尾转换。以下检查可验证克隆后
归档和补丁的字节完整性；它不代替官方 runtime 的来源 preflight 或训练验收：

```bash
python research/rl_correctness_20260907/verify_archive.py
git fsck --full
```

本次维护前的原始 Git bundle、未提交工作树压缩包，以及维护后的恢复 bundle，均有
独立副本。首次推送提交 `79abe153d5368eccca81a9362c156756e2e24741` 已通过服务器
现有 GitHub SSH 权限保存并核对远端 ref。凭据没有写入仓库。无需依赖单一工作树
才能恢复代码，但 Git 的代码备份不等于模型与大型实验数据的异地备份。
