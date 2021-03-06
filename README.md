# AR應用 - 多少航航出現在鏡頭上
## 2021/06/20 完稿  
### 翁健愷
###### tags: `computer vision`

## 以下演算法大多皆為手刻
main_final.py 裡可以選擇要偵測特徵照片，以及要貼上的影片  !!! 影片要先自行轉成 npy 檔 !!!

prospective.py 裡主要以手刻 perspective transformation，還有抓取特徵照片角落、影片角落演算法為主

featurepoint.py 裡主要以找尋兩張圖的特徵點，比對並過濾出好的特徵點，再去進行應用跟查看。並且有最後的貼上影片的 function 在這裏面

## 簡介
專題用 opencv 搭配鏡頭去找尋目標圖片，並將目標圖片替換成影片，使用者可以觀測圖片以及影片，找出有多少航航。<font color = 'red'>**答案自在人間(X**</font>

## 簡單描述大致過程
先對兩張圖片找取其特徵點，比對找特徵點後，找出好的特徵點後，**自動的**根據目標圖與背景圖的的四個點，進行perspective transformation。

* 附圖為找到好的特徵點
![](https://i.imgur.com/z07OXJG.jpg)

如果今天貼得好的話，我再將目標圖片的四個角落乘上 transformation matrix，就應該貼到背景圖裡的特徵圖的角落，再根據這個角落，將我們新的照片、影片進行貼上。

* 抓背景圖中的目標圖角落概念(先透過黑色找到轉換矩陣，再將橘色的點帶入同樣的轉換矩陣)
![](https://i.imgur.com/KZaICIv.png)

用鏡頭的話，就得根據鏡頭抓取進來的照片，馬上去做運算，在特徵圖片移動時也要能追蹤特徵圖片，將影片貼上，要讓影片能順利的運行，還需要進行一些優化以及思緒上的建構。

* 我們讓特徵照片移動，影片也需要跟著移動

點下方圖片會跑到影片地方

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/ExOWR5VREk8/0.jpg)](https://www.youtube.com/watch?v=ExOWR5VREk8)

或是這裡有網址[針對單一照片搜尋並替換成照片](https://www.youtube.com/watch?v=ExOWR5VREk8)

## 針對單一照片搜尋並替換成照片
這邊還沒有進行演算法的優化，因次當時很容易浮動，而且當時是用手持攝影機，因此抖動可能會影響判斷過程。
點下方圖片會跑到影片地方
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/4yoJXJ-W3Xk/0.jpg)](https://www.youtube.com/watch?v=4yoJXJ-W3Xk)

或是這裡有網址[針對單一照片搜尋並替換成照片](https://www.youtube.com/watch?v=4yoJXJ-W3Xk)

## 針對單一照片搜尋並替換成影片
這時已經有優化，降低了貼歪的機率，雖然有時還是會爆掉QQ，還要再想辦法處理
點下方圖片會跑到影片地方
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/EoXiZ289wo4/0.jpg)](https://www.youtube.com/watch?v=EoXiZ289wo4)

或是這裡有網址[針對單一照片搜尋並替換成影片](https://www.youtube.com/watch?v=EoXiZ289wo4)

## 最終影片
找到三張特定照片，都去進行更換成影片，本來最後想弄[另外一個影片的](https://www.youtube.com/watch?v=Tf-E5oKYvvE)<font color = 'red'>**這樣根本數不出有幾個統神XD**</font>，但出現了這個，Memory error，好像存太多 np 的檔案，已經超出範圍所以不給存了。好慘QQ

![](https://i.imgur.com/lIuEYcu.png)

所以最後只能將原本三個影片再拿來播放XD，還有精美音效喔!! >_Ob
點下方圖片會跑到影片地方
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/J2KVbKkFPXk/0.jpg)](https://www.youtube.com/watch?v=J2KVbKkFPXk)

或是這裡有網址[最終影片](https://www.youtube.com/watch?v=J2KVbKkFPXk)

## 心得
這次我沒選擇做 unity 以及另一個，但這個動態照片，其實也不簡單，一開始很常找不到好的點，因此會高機率沒貼好，我後來在找尋好的轉換點時，因為我們要找的特徵圖，應該會在背景圖裡的某個區域，因此我嘗試用統計學的概念，過濾掉離群值，讓計算量變少，並去除四個轉換點靠太近的情況，貼成功的機率也就大大的上升了。

而且這次在測試過程中，我寫了很多可視化來觀察點、圖、影片到底怎麼樣，實在是大大解決很多問題，但也因為這樣發現很多問題，影片跑太慢、點不知道找去哪了......，寫期末 project 雖然沒有寫很好，但也盡力了。
