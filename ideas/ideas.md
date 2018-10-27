#問題の理解
広告をクリックするのが本物のユーザーか詐欺かを当てる、ダウンロードしたかを当てる？
問題：click_idとis_atributed(appをダウンロードしたかどうか？)
評価指標：AUC

#Data
ip: クリックしたipアドレス
app: アプリのid
device: デバイスのタイプ
os: osのバージョン
click_time: クリックした時間
channel id: channel id of mobile ad publisher
attributed_time: もしユーザーが広告をクリックしたあとにアプリをダウンロードしたら、その時刻
is_attributed: アプリがダウンロードされたかどうか

#データのロード
ひとまずデータをcsvから普通にpd.read_csv()してみる。
trainとtest合わせてcsvだと8.5Gくらい。

#基本的にpandasだと遅すぎる。
もろもろ試してみるのに小さいデータセット作るか。
全体EDAはBQ、可視化諸々はpandasでできたら理想。
小さいデータ・セットで試して大丈夫かどうかは、時系列で変化が無いか確認をする。

##Some DEA
・TrainとTestの時系列でのデータ数
・特にattributed_time
・cｓ
