import time
from selenium import webdriver
from bs4 import BeautifulSoup
import os, requests, re

# 获取当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 图片存储地址
path = os.path.join(current_dir, 'baidu_images')
# 正则表达式 过滤并得到图片标题中所有的中文
pattern = '[\u4e00-\u9fa5]+'
# 最大下载图片数量
max_images = 100


def parseHtml(text, downloaded_count):
    # 因为title标签有很多错误数据，所以记录一下下标
    titleIndex = 0
    # HTML解析器
    b = BeautifulSoup(text, "html.parser")
    # 得到页面中所有 img标签并且class是main_img img-hover开头的图片列表
    img = b.find_all("img", class_="main_img img-hover")
    # 得到页面中所有 a标签并且class是imgitem-title开头的图片标题列表
    title = b.find_all("a", class_="imgitem-title")
    # 循环拿到的图片列表，解析标题下载图片
    for index, value in enumerate(img):
        if downloaded_count >= max_images:
            return downloaded_count
        imageUrl = img[index]["data-imgurl"]
        imageName = "default" + str(index)
        # 一个循环用于拿到每个图片的正确标题
        if len(title) > index:
            titleIndex = getImageTitle(title, titleIndex)
            imageName = "".join(re.findall(pattern, title[titleIndex].text))
            if imageName == "":
                imageName = "default" + str(index)
            titleIndex += 1
        imageName += ".jpg"
        print("准备下载第%s个图片，名字叫：%s 地址：%s" % (str(downloaded_count + 1), imageName, imageUrl))
        savePhoto(imageUrl, imageName)
        downloaded_count += 1
    return downloaded_count


# 页面里有很多class=imgitem-title的标签，图片的标题一般有title属性，根据title属性过滤标签
def getImageTitle(title, index):
    if "title" in title[index].attrs:
        return index
    else:
        getImageTitle(title, index + 1)
        return index + 1


# 保存照片
def savePhoto(imageUrl, filename):
    p = os.path.join(path, filename)
    c = requests.get(imageUrl)
    createFile(path)
    with open(p, 'wb') as f:
        f.write(c.content)
    time.sleep(0.3)
    print("下载完了 休息0.3秒")


# 创建本地文件
def createFile(path):
    print("path:", path)
    file = os.path.exists(path)
    if not file:
        os.makedirs(path)


# 滑动浏览器窗口，加载更多图片，滑动一次加载20多张图片
# number为想滑动的次数
def slideBrowseWindow(driver, number):
    for i in range(number):
        time.sleep(0.1)
        driver.execute_script("window.scrollBy(0,{})".format(i * 1000))
        time.sleep(0.3)


if __name__ == "__main__":
    # 这里的豪车可以改成自己想搜索的内容
    searchName = "00后女明星大全"
    # 这里使用的是edge浏览器，如果想使用谷歌浏览器，可以换成webdriver.Chrome()
    driver = webdriver.Chrome()
    driver.get(
        "https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1709260324755_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&dyTabStr=MCwxLDMsMiw4LDYsNCw1LDcsOQ%3D%3D&ie=utf-8&sid=&word=" + searchName + "&f=3&oq=%E7%BE%8E%E5%A5%B3&rsp=0")
    # 百度图片是下滑加载更多
    downloaded_count = 0
    while downloaded_count < max_images:
        slideBrowseWindow(driver, 1)
        # 获取页面的源代码
        page_source = driver.page_source
        # 输出页面源代码
        downloaded_count = parseHtml(page_source, downloaded_count)
    driver.quit()
    print("done.")