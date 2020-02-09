## 备份
新建branch，设置为default

更改文件后（记得更新.gitignore）
```bash
git add .
git commit -m "note"
git push
```


## 恢复
安装git、node js、npm
npm install git 
设置git全局邮箱和用户名
```bash
git config --global user.name "yourgithubname"
git config --global user.email "yourgithubemail"
```

设置ssh key
```bash
ssh-keygen -t rsa -C "youremail"
#生成后填到github
#验证是否成功
ssh -T git@github.com
```

安装hexo（不需要初始化）
```bash
npm install hexo-cli -g
```

clone repository, 安装hexo-deployer-git
```
git clone ...
npm install hexo-deployer-git --save
```
