#! /bin/bash
#set -x
function read_dir()
{

for file in `ls $1` #注意此处这是两个反引号，表示运行系统命令
do
    # echo $1"/"$file
    if [ -d $1"/"$file ] #注意此处之间一定要加上空格，否则会报错
    then
        read_dir $1"/"$file
    else
        echo $1/$file #在此处处理文件即可
        flie2=$1/$file
        echo "图像的地址为："$1/$file >>output.txt
        python3 Human-face-gesture.py $flie2  >>output.txt
        echo '\n'  >>output.txt
    fi
done
} 


if [[ output.txt ]]; 
then
    rm output.txt
fi
#读取第一个参数
read_dir $1
