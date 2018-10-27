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
attributed_time: もしユーザーが広告をクリックしたあとにアプリをダウンロードしたら、その時刻
is_attributed: アプリがダウンロードされたかどうか

#データのロード
ひとまずデータをcsvから普通にpd.read_csv()してみる。
trainとtest合わせてcsvだと8.5Gくらい。
