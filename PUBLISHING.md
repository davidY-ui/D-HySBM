# 上传 GitHub 与生成论文引用

这份副本已排除大规模模拟、无关实验输出和编译缓存，并保留五个 Notebook 实际重画图所需的小型原始输入于 `plot_data/`。原项目未被改写。请在此发布副本中操作。

## 上传之前

1. 阅读 README 中的架构来源说明；保留 Ruggeri 等人的引用与上游许可证。
2. `CITATION.cff` 的代码作者按作者指定填写为 Dawei Ye。论文作者在 README 中单独列出。
3. 为本项目新增贡献选择合适的许可证，再补充根目录 `LICENSE` 与 CFF 的 `license` 字段；这里没有代为选择许可证。
4. 先按 `docs/NOTEBOOKS.md` 重画图并运行检查工具；本包包含绘图所需汇总、拟合与测试输入。完整训练/模拟复现还需要相应完整输入及参数，并应填写真实数据存档地址。BP 单代缓存明确为历史 30k-path 结果，不要写成新的 200k-path 实验。

## 创建仓库和推送

在 GitHub 创建公开空仓库，例如 `D-HySBM`，不自动添加 README、许可证或 `.gitignore`。在解压后的项目根目录打开 PowerShell：

```powershell
git init -b main
git config user.name "Dawei Ye"
git config user.email "YOUR_COMMIT_EMAIL"
git add .
git status
```

核对待提交文件，再把下面的 `YOUR_USERNAME` 替换为实际 GitHub 用户名并执行：

```powershell
git commit -m "Prepare source release accompanying the paper"
git remote add origin https://github.com/YOUR_USERNAME/D-HySBM.git
git push -u origin main
```

这里的用户名、邮箱是说明用占位符，不是已经配置的远程仓库或作者邮箱。

## Zenodo 与 Release

先在 Zenodo 连接 GitHub 账户，在 GitHub 页面 `Sync now` 后开启本仓库，再在 GitHub 创建版本标签与 Release（例如首次正式版本 `v1.0.0`）。等 Zenodo 归档处理成功后，核对记录中的标题、作者、版本与实际文件，并取得该版本 DOI。

随后在 CFF 补充真实 `repository-code`、`version`、`date-released` 和 `doi`，在 README/论文写入真实链接。最初的 DOI 产生于归档后；不要为补入它而覆盖已经发布的版本标签。对后续代码更新创建新的版本。

论文中引用用于产生结果的具体版本 DOI。Zenodo 的 Concept DOI 代表全部版本，并指向最新版本。

官方说明：[GitHub 上传](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=windows)、[启用 Zenodo 仓库](https://help.zenodo.org/docs/github/enable-repository/)、[DOI 版本](https://zenodo.org/help/versioning)。
