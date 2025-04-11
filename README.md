# 复旦自动刷锻脚本

使用光华智慧体育小程序的 API 实现自动刷锻。

**GitHub Actions 因为运行时间过长，占用服务器资源，因此被禁用。请本地运行此脚本。Python 安装教程和命令行配置教程待完善**

## 使用

- 安装依赖：`pip install -r requirements.txt`
- 修改 `settings.json` 文件中的 `USER_ID, FUDAN_SPORT_TOKEN` 变量，需要在小程序内抓包获得，详请查看“抓包教程”章节。
- 查看刷锻路线列表：`python main.py --view`
- 自动刷锻：`python main.py --route <route_id>`，其中 `route_id` 是刷锻路线列表中的 ID。
- 可以设置里程和时间，如 `--distance 1200 --time 360`，更多选项请使用 `python main.py --help` 查看。
- （附加）环境变量 `PLATFORM_OS, PLATFORM_DEVICE` 可以设置刷锻的设备标识，默认值为 `iOS 2016.3.1`
  、`iPhone|iPhone 13<iPhone14,5>`。

## 说明

目前支持的场地有菜地、南区田径场和江湾田径场。

## 抓包教程

#### iOS 系统

抓包教程可参考 [使用 Stream 抓包](https://www.azurew.com/%e8%bf%90%e7%bb%b4%e5%b7%a5%e5%85%b7/8528.html)
，抓包软件可在 [App Store](https://apps.apple.com/cn/app/stream/id1312141691) 下载。

按照教程内的指引配置到设置证书的步骤，然后在软件内点击 Sniff Now 按钮，打开刷锻小程序刷新一下（确保小程序已经登录），再回到
Stream，点击 Stop Sniffing，然后点击 Sniff
History，选择最近的一条记录，点开后找到开头为 `GET https://sport.fudan.edu.cn/sapi` 的任意一条记录，点进去选择 Request，在
Request Line 中有 `userid=xxx&token=xxx` 的记录，记下这两段信息。

#### Windows 系统

可参考 [教程](https://juejin.cn/post/6920993581758939150/) 进行相应设置。注意：需要把 Fidder 中 HTTPS 部分设置的复选框由
from browers only 改为 from all processes。

在配置完后，微信登录，右上角齿轮进入代理，端口为 127.0.0.1，端口号为 8888（默认）
登录后进入小程序并登录，在 fiddler 里找到下图中的 ID 和 token
![image](https://user-images.githubusercontent.com/51439899/226794395-42eca333-fb65-4e29-a2cb-b8ce3fd13221.png)

**注意，目前 Token 的有效期为 3 天。**

### Issue

如果在使用过程中遇到了问题，请点击页面顶部的 Issue - New Issue，并在出现的文本框中描述你的问题。
