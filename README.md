# 2024全國跨領域高階人才培育PITCH內容
## 研究題目: 
    皮膚瘤遠端監測與輔助治療系統
## 研究動機: 
根據國民健康署公佈的108年資料，皮膚癌為發生率上升最快的五大癌症，值得我們多加注意，因此，及早檢測和診斷變得越來越重要。目前，皮膚癌的診斷在台灣以切片為大宗，而在先進國家則是以非侵入式的治療為主，然而，傳統的診斷過程耗時且昂貴，因此，自動化、準確的皮膚癌分類和病變分割方法顯得尤為重要。基於深度學習的影像處理技術已被證實在醫學影像分析中具有卓越的效果，使用卷積神經網絡（CNN）進行皮膚病變的分割和分類成為近年來研究的趨勢所向，所以我們在這份研究中也嘗試利用深度學習模型來判斷皮膚癌的種類，期望能以此達到降低人力成本做為一種輔助醫療的手段，而皮膚癌疾病還有一個問題是需要大量的追蹤才能有效掌握病情的發展，但受限。而嵌入式系統的發展為這件事帶來無限的可能性，因此，我們希望可以通過NVIDIA開發的JETSON NANO開發板作為移動式的裝置能讓我們實現遠端智慧醫療的願景。
## 研究方法:
我們的資料集來源為ISIC201皮膚癌疾病資料集，裡面包含三種不一樣的疾病狀態包含: Seborrheic keratosis、Melanoma、Navis，我們計畫透過mobile101、VGG19、RESNET18與自製簡易CNN模型嘗試分辨出三種不同的疾病，並將推論程式部屬在JETSON NANO開發板上，然後利用FLASK網頁腳本將手機鏡頭拍攝到的皮膚畫面透過開發板進行實時預測
## 設備與計畫流程:

![image](https://github.com/user-attachments/assets/fddea05c-47d8-40cb-861c-2f3424c95d1b)

圖一.簡要說明(設備與我們想達成的目標)

![image](https://github.com/user-attachments/assets/4f1d37c3-8fd2-439d-82a8-daa901aa077a)
	 
圖二.研究計畫流程圖




## 硬體與軟體架構流程圖設計:

![image](https://github.com/user-attachments/assets/1d1bcefa-318b-438c-a0df-d8fc07441321)


 
圖三.硬體架構設計

![image](https://github.com/user-attachments/assets/f7311af4-8ca9-466b-9393-ef56a1d16c7a)

圖四.軟體架構流程圖

![image](https://github.com/user-attachments/assets/952b5b16-20de-4c1a-bd7b-538decb58223) 
![image](https://github.com/user-attachments/assets/8e024d6b-bb01-4feb-9a31-3e2311c5200a)

 
圖五.深度學習軟體架構設計(運用RESNET18模型對兩種皮膚癌疾病分別進行預測)

## 研究結果：
實作後的結果發現，確實可以達成實時將皮膚癌影像上傳的功能，達成遠距智慧醫療的能力，而深度學習推論三種不同的疾病也有一定的準確性，約可以達到70%的準確度，雖然說無法完全取代醫生的判斷，但作為一個大眾之間輔助的智慧醫療儀器已十分足夠，而未來我們也希望可以往增加準確度這一個方向邁進，尋找更有有效的模型作為我們的推論程式。

