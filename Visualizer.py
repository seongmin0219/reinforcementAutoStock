import threading
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from mplfinance.original_flavor import candlestick_ohlc
from agent import Agent


lock = threading.Lock()

class Visualizer:
    COLORS = ['r','b','g']


    def __init__(self, vnet=False):
        self.canvas = None
        #캔버스 같은 역할을 하는 Matplotlib의 Figure 클래스 객체
        self.fig = None

        #차트를 그리기 위한 Matplotlib의 Axes 클래스 객체

        self.axes = None
        self.title = '' #그림 제목


        def prepare(self, chart_data , title):
            self.title = title

            # 다른 스레드 접근 못한 상태에서 진행
            with lock:
                # 캔버스를 초기화하고 5개의 차트를 그릴 준비
                # figure 안에 axes 가 하나 이상 포함되어 있는 구조 fig는 캔버스 axes는 하나 하나의 그림
                # 5행 1열이므로 5개의 그림을 1열로 가지는 캔버스

                self.fig, self.axes = plt.subplots(
                    nrows=5 , ncols=1, facecolor = 'w', sharex=True)

                for ax in self.axes:
                    # 보기 어려운 과학적 표기 비활성화
                    ax.get_xaxis().get_major_formatter() \
                        .set_scientific(False)
                    ax.get_yaxis().get_major_formatter() \
                        .set_scientific(False)
                    # y axis 위치 오른쪽으로 변경
                    ax.yaxis.tick_right()
                # 차트 1. 일봉 차트
                self.axes[0].set_ylabel('Env.') # y축 레이블 표시
                x = np.arange(len(chart_data))
                # open , high, low, close 순서로 된 2차원 배열
                ohlc = np.hstack((
                    x.reshape(-1,1),np.array(chart_data)[:,1:-1]))
                #양봉은 빨간색으로 음봉은 파란색으로 표시

                #candlestick_ohlc 입력은 ohlc와 axes 객체 하나
                # colorup = 양봉
                candlestick_ohlc(
                    self.axes[0], ohlc, colorup='r', colordown='b'
                )

                # 거래량 가시화
                ax = self.axes[0].twinx()
                volume = np.array(chart_data)[:,-1].tolist()
                ax.bar(x,volume, color='b', alpha=0.3 )

    def plot (self , epoch_str=None, num_epochs=None, epsilon=None,
              action_list=None, actions=None, num_stocks=None,
              outvals_value=[], outvals_policy=[], exps = None
              , learning_idxes=None , initial_balance=None, pvs=None):
        with lock:
            x= np.arange(len(actions))
            actions = np.array(actions) #에이전트의 행동 배열
            # 가치 신경망의 출력 배열
            outvals_value = np.array(outvals_value)
            # 정책 신경망의 출력 배열
            outvals_policy = np.array(outvals_policy)
            # 초기 자본금 배열
            # 2차원 이상 시 차원을 튜플로
            pvs_base = np.zeros(len(actions)) + initial_balance
            #zip 은 두 배열에서 같은 인덱스로 묶는다

            # 가능한 두 리스트까지.

            # 차트 2. 에이전트 상태 (행동 ,보유 주식 수)
            # zip은 tuple 형태로 두 리스트를 묶는다
            for action , color in zip(action_list, self.COLORS):
                for i in x[actions == action]:
                    # 배경색으로 행동 표시
                    self.axes[1].axvline(i, color=color, alpha =0.1)
            # x y 축 길이가 같은 데이터
            self.axes[1].plot(x, num_stocks , '-k') # 보유 주식 수 그리기

            # actions : 에이전트가 수행한 행동 배열
            # action_list 에이전트가 수행가능한 전체 행동 리스트
            #  차트 3 . 가치 신경망
            # 행동에 대한 예측 가치를 라인 차르로 표현
            # outvals_value 가치 신경망 출력 배열
            if len(outvals_value) > 0:
                # y축일 경우 axis=1 이고 값이  아니라 인덱스를 반환한다
                max_actions = np.argmax(outvals_value, axis=1)
                for action , color in zip(action_list, self.COLORS):
                    # 배경 그리기
                    for idx in x:
                        if max_actions[idx] == action:
                            # axes는 캔버스 속 하나의 그림
                            self.axes[2].axvline(idx,color=color, alpha =0.1)
                        # 가치 신경망 출력의 tanh 그리기
                        self.axes[2].plot(x, outvals_value[:,action],
                                          color=color, linestyle='-')

            # 차트 4 정책 신경망
            # 탐험을 노란색 배경으로 그리기
            for exp_idx in exps:
                self.axes[3].axvline(exp_idx, color= 'y')
            # 행동을 배경으로 그릭
            # pep 8 _ priavte 변수
            _outvals = outvals_policy if len(outvals_policy) > 0 \
                else outvals_value
            for idx, outval in zip(x, _outvals):
                color = 'white'
                if np.isnan(outval.max()):
                    continue
                if outval.argmax() == Agent.ACTION_BUY:
                    color = 'r' # 매수 빨간색
                elif outval.argmax() == Agent.ACTION_SELL:
                    color = 'b' # 매도 파란색
                self.axes[3].axvline(idx, color=color, alpha=0.1)
            # 정책 신경망의 출력 그리기




