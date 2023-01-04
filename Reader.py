import pygame
from pygame.locals import *
import torch
import torch.nn as nn   # ニューラルネットワークのモジュール
import torch.optim as optim
import time


from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations,WindowOperations

# ボタンは 0から11まで
# アナログボタンを押すと joyaixisモーションがアナログで動く
# 軸0 左アナログコントローラ左右(左が-1から1)
# 軸1 左アナログコントローラ上下(上が-1から下が1)
# 軸2 右アナログコントローラ左右(左が-1から下が1)
# 軸3 右アナログコントローラ上下(上が-1から下が1)
# プログラム(e.button) ジョイスティックの番号　位置
# 0 1ボタン
RIGHT_ONE_BUTTON = 0
# if e.button == RIGHT_ONEBUTTON :
#    print("1 button pushed");
# 1 2ボタン
RIGHT_TWO_BUTTON = 1
# 2 3ボタン
RIGHT_THREE_BUTTON = 2
# 3 4ボタン
RIGHT_FOUR_BUTTON = 3
# 4 5ボタン　左上トリガー
LEFT_UP_TRIGGER = 4
# 5 6ボタン  右上トリガー
RIGHT_UP_TRIGGER = 5
# 6 7ボタン　左下トリガー
LEFT_DOWN_TRIGGER = 6
# 7 8ボタン　右下トリガー
RIGHT_DOWN_TRIGGER = 7
# 8 9ボタン リセットボタン
RESET_BUTTON = 8
# 9 9ボタン スタートボタン
START_BUTTON = 9
# 10 左アナログボタン
LEFT_ANALOG_BUTTON = 10
# 11 右アナログボタンスイッチ
RIGHT_ANALOG_BUTTON = 11


class JoyStickReader():
    def __init__(self,bm) :
        self.bm = bm
        self.trainMode = True
        pygame.joystick.init()
        try:
            self.j = pygame.joystick.Joystick(0)
            self.j.init()
        except pygame.error:
            print('ジョイスティックが接続されていません')
        pygame.init()
        BoardShim.enable_dev_board_logger()

        # use synthetic board for demo
        self.params = BrainFlowInputParams()

        # params.serial_port = args.serial_port
        self.params.serial_port = "COM11"
        #params.other_info = args.other_info
        #params.serial_number = args.serial_number
        #params.ip_address = args.ip_address
        #params.ip_protocol = args.ip_protocol
        # params.timeout = args.timeout
        self.board_id = BoardIds.CYTON_DAISY_BOARD.value
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.board = BoardShim(self.board_id, self.params)
        self.board.prepare_session()
        


    def readJoyStick(self):
        while True:
            for e in pygame.event.get():  # イベントチェック
                if e.type == QUIT:  # 終了が押された？
                    self.board.release_session()
                    return
                if (e.type == KEYDOWN and
                        e.key == K_ESCAPE):  # ESCが押された？    
                    self.board.release_session()
                    return
                # Joystick関連のイベントチェック
                if e.type == pygame.locals.JOYAXISMOTION:  # 7
                    x, y = self.j.get_axis(0), self.j.get_axis(1)
                    z, t = self.j.get_axis(2), self.j.get_axis(3)
                    print('x and y : ' + str(x) + ' , ' + str(y))
                    print('z and t : ' + str(z) + ' , ' + str(t))
                elif e.type == pygame.locals.JOYBALLMOTION:  # 8
                    print('ball motion')
                elif e.type == pygame.locals.JOYHATMOTION:  # 9
                    print('hat motion')
                elif e.type == pygame.locals.JOYBUTTONDOWN:  # 10
                    self.pressedButton(e.button)
                    print(str(e.button)+'番目のボタンが押された ','train mode %d' % self.trainMode )
                elif e.type == pygame.locals.JOYBUTTONUP:  # 11
                    self.releasedButton(e.button)
                    print(str(e.button)+'番目のボタンが離された')
            if(self.trainMode==False):
                self.board.start_stream()
                time.sleep(1)
                self.board.stop_stream()
                data = self.board.get_board_data(num_samples=32*60)
                print(data)
                print(data.shape)
                data = data[:,0:60]
                print(data.shape)
                self.bm.eval()
                exp = self.bm.forward(torch.tensor(data.reshape(1,32*60),dtype=torch.float32))
                print(exp.max(1)[1].item())


    '''
    # ボタン押されたら学習スタート
    # ボタンは right_one_buttonの場合は↑に移動の学習

    '''
    def pressedButton(self,button):
        print("button " + str(button) + " is pressed ")
        if button == RESET_BUTTON :
            self.trainMode = True
        elif button == START_BUTTON:
            self.trainMode = False
        else:
            if(self.trainMode==True):
                self.trainNode(button)


    def trainNode(self,button):
        optimizer = optim.Adam(self.bm.parameters(), lr=self.bm.learning_rate)
        criterion = nn.CrossEntropyLoss()
        num_epochs = 5
        optimizer.zero_grad()
        # data from bci 
        self.board.start_stream()
        time.sleep(1)
        self.board.stop_stream()
        data =self.board.get_board_data(32*60)
        data = data[:,0:60]
        self.bm.train()
        print(torch.tensor(data).shape)
        outputs = self.bm(torch.tensor(data.reshape(1,32*60),dtype=torch.float32));
        labels = []
        labels.append(int(button))
        labels =torch.tensor(labels,dtype=torch.long)
        print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        

    '''

    # comment
    # ボタン離したときは学習ストップ
    #
    #
    '''

    def releasedButton(self,button):
        print("button " + str(button) + " is released ")

    def train(self):
        self.bm.train()
        optimizer = optim.Adam(self.bm.parameters(), lr=self.bm.learning_rate)
        criterion = nn.CrossEntropyLoss()
        num_epochs = 5
