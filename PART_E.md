# Part E 上機手冊

## 老師的兩檔部署方式

若已依講義建立並註冊 send_script 套件，把 repo 內 src/hw3_robot/hw3_robot/ 的 image_sub.py 與 send_script.py 複製到 ~/workspace2/teamX_ws/src/send_script/send_script/，覆蓋兩支範例。在 send_script.py 的 LAB 字典填寫下方校正值，或用 calibration 參數提供外部 JSON。兩檔都提供 main()，只需實驗室 ROS、cv_bridge、OpenCV、NumPy 和 tm_msgs。回 teamX_ws 重新 colcon build 並 source install/setup.bash。

依序啟動 RosNode、老師的 tm_driver、tm_get_status image_talker，再分終端執行 ros2 run send_script image_sub 和 ros2 run send_script send_script。預覽時在固定姿態另外觸發 job1；實抓加 --ros-args -p execute:=true。下方 hw3_robot 是使用本 repo 套件時的名稱。
## 資料流與座標

```text
TMflow job1 → techman_image (sensor_msgs/Image) → detect_blocks
    → detect_blocks/result (std_msgs/String，JSON) → stack_blocks
TM driver → feedback_states (tm_msgs/FeedbackState) → stack_blocks
stack_blocks → send_script (tm_msgs/SendScript) → TM Listen → TCP 移動、拍照
stack_blocks → set_io (tm_msgs/SetIO) → 工具端 DO 0 → 夾爪
```

相機固定在手腕上，每輪在相同 TCP 姿態拍一張照片。辨識節點對灰階影像以 50 二值化，保留面積至少 90 px² 的輪廓。每個物件輸出重心與最小外接矩形長邊的方向點。控制節點對兩點分別做平面單應轉換，再在底座 XY 平面求角度。此法適用於等高、平放且分開的積木；換相機姿態、TCP、解析度或積木高度須重校。C 的深度圖與 A 的自選相機內參都不是此流程的輸入。

## 1　準備設備與 ROS

依實驗室程序與 TM5 Guide Book 的 Safety Information、Emergency Stop、Tool End I/O Interface 確認急停、工具接線及操作區。由現場人員確認手臂型號、TCP、負載、Base、速度限制及空間。清空移動路徑，第一次低速驗證。PTP 為關節插補，兩個端點都安全仍需檢查中途掃過的空間。程式逾時會停止發下一筆；已送出的移動需用控制器停止。

在控制器啟動實驗室提供的 Listen 專案、相機工作 job1，以及對應機型的 ROS driver 和影像發布器。各終端都先 source 系統 ROS、實驗室 driver 工作區，再 source 本工作區。以下從 repo 根目錄執行：

```bash
source /opt/ros/humble/setup.bash
# 此處 source 實驗室 TM driver 的 install/setup.bash
colcon build --packages-select hw3_robot
source install/setup.bash
ros2 service type /send_script
ros2 service type /set_io
ros2 topic info -v /techman_image
ros2 topic echo /feedback_states --once
```

核對服務型別與上圖相符。確認 tool_pose 對應示教器目前 TCP，位置回授是公尺、轉角是弧度；送出的 CPP 為毫米、度。A/B/C 分別繞 X/Y/Z；程式用 Rz(C) Ry(B) Rx(A) 比對到位。用示教器低速確認工具 DO 的 1 為閉、0 為開。相機時間戳要與 ROS 時鐘一致。

## 2　取得現場校正值

複製 `lab.example.json` 為 `lab.json`，逐項填入現場量測值。先用示教器移至 e/send_script.py 的指定拍照姿態，右臂 [230,230,730,-180,0,135]，左臂 [350,350,730,-180,0,135]，核對機台後填入 camera_pose。觸發 job1，以 rqt_image_view 選 techman_image 檢查光線、視野、解析度，填 image_size 為 [寬,高]。

在積木頂面高度設置至少四個分散、非共線的標記。從同一拍照姿態記錄每個標記中心 [u,v] 到 pixels；示教取得該點在底座的 [X,Y] 毫米，依相同順序填 base_xy_mm。取額外標記檢查預測與實測偏差小於夾爪可容許的偏心。所有候選積木與方向點須落在校正區內。

手動示教單塊中心的抓取 TCP 高度 grasp_z_mm，量 block_height_mm。選一個遠離拾取區的空位作 stack_xy_mm。clear_z_mm 須足以讓**夾著的積木**通過三層堆疊和障礙物。以一塊已知方向的積木示教，量出邊線方向到夾爪 C 角的修正 jaw_offset_deg，再記錄堆放方向 stack_yaw_deg 與夾爪開閉穩定所需 gripper_settle_s。所有高度共用相同 Base、TCP。

## 3　預覽與實機

終端一啟動辨識，終端二啟動下列預覽；接著從控制器在已校正姿態拍攝新圖：

```bash
ros2 run hw3_robot image_sub
# 另一個已 source 的終端
ros2 run hw3_robot send_script --ros-args -p calibration:="$PWD/lab.json"
```

預覽只列出 IO 和 TCP 指令。用 `ros2 topic echo /detect_blocks/result --once` 核對恰有三個中心，各自落在不同積木。若桌面也被辨識，先改用深色霧面底板、均勻照明，再重拍。若逾時，檢查 topic、job 名稱與 header.stamp。比對預覽的每一組座標、角度、堆放高度及手臂可達範圍。先用示教器在積木上方核對 XY、夾爪方向，再檢查下降與離開路徑。

確認無人進入工作範圍，由現場操作員掌握停止裝置，從已驗證的起始姿態執行：

```bash
ros2 run hw3_robot send_script --ros-args -p calibration:="$PWD/lab.json" -p execute:=true
```

控制節點移至拍照姿態，觸發 job1，等待新圖，產生整輪計畫。每塊依序靠近、下降、閉爪、升起、移至堆疊區、下降、開爪、升起。第 2、3 層的放置高度各增加一個塊高。移動後以回授確認到位，IO 以服務回覆與實測等待時間同步。無夾持感測；若落空、滑落、積木被推動或出現異常，立即依現場程序停機，重置場景再重新拍照。正式錄影從空桌開始，拍到手動隨機撒入三塊及完整堆疊過程。

## 重現測試與調整

從本工作區根目錄，在已 source 的終端執行。模擬測試須選擇沒有其他節點的專用 domain，在本機模擬服務上執行。

```bash
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=187
python3 tests/part_e_preview.py
# 已建置 tm_msgs 時，可用模擬控制器測試完整服務流程
python3 tests/part_e_preview.py --mock-tm
```

回到實機時使用新的終端並恢復實驗室的 ROS domain 與網路設定。辨識漏掉深色積木時可降低 threshold；桌面雜點太多時提高 min_area，大面積背景則降低 max_area。請先用實拍影像估計積木面積，例如用以下語法覆寫節點預設值，再重跑預覽。

```bash
ros2 run hw3_robot image_sub --ros-args -p threshold:=50 -p min_area:=90 -p max_area:=1000000
```

本次在 Ubuntu-D、ROS 2 Humble，以 Techman 官方 tmr_ros2 的 humble 分支（c40bfc0）建置 tm_msgs。模擬測試驗證拍照姿態、job1、三輪抓放、三個堆疊高度與七次 IO；模擬器直接回報指令目標，未模擬路徑、接觸或夾力。[官方驅動與啟動說明](https://github.com/TechmanRobotInc/tmr_ros2/tree/humble)

