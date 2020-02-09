---
title: 常用conda命令
date: 2018-11-14 14:25:15
toc: true
category: Tech Zoo
tags:
---
本文记录了常用的conda命令，以供查询。

<!--more-->

## 基本操作
| 描述                       | 命令                                      |
| -------------------------- | ----------------------------------------- |
| 获取conda版本号            | `conda --version`                         |
| 查看当前所有环境           | `conda info -e`                           |
| 创建新环境(指定python版本) | `conda create -n name1 python=3.7`        |
| 创建包含某些包的环境       | `conda create -n name1 numpy scipy`       |
| 激活环境                   | `source activate name1`                   |
| 退出当前环境               | `source deactivate `                      |
| 复制某个环境               | `conda create --name name2 --clone name1` |
| 删除某个环境               | `conda remove --name name1 --all`         |

## 包管理
一般直接使用pip安装包。


| 描述                 | 命令                  |
| -------------------- | --------------------- |
| 列举当前环境的所有包 | `conda list`          |
| 列举制定环境的所有包 | `conda list -n name1` |

pip换源安装包  
`pip install scrapy -i https://pypi.tuna.tsinghua.edu.cn/simple`


## 分享环境
一个分享环境的快速方法就是提供你的环境的.yml文件。

- 首先通过activate target_env要分享的环境target_env，然后输入下面的命令会在当前工作目录下生成一个environment.yml文件
- `conda env export > environment.yml`
- 小伙伴拿到environment.yml文件后，将该文件放在工作目录下，可以通过以下命令从该文件创建环境
- `conda env create -f environment.yml`
- .yml实际是一个将环境中所有包的版本在孤独办法的结构化文本文件

## conda的维护
| 描述 | 命令                                      |
| ---- | ----------------------------------------- |
| 升级 | `conda conda update conda`                |
| 删除 | `conda rm -rf ~/anaconda（直接删除目录）` |
| 帮助 | `conda remove -h`                         |



> 参考资料：https://blog.csdn.net/yimingsilence/article/details/79388205