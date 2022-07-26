import streamlit as st
from streamlit_option_menu import option_menu
import torch.optim.lr_scheduler as lrs
import matplotlib.pyplot as plt
from io import BytesIO

from model import BasicModel
from utils import plot_lr, LambdaLR, MultiplicativeLR, \
                  StepLR, MultiStepLR, ExponentialLR,  \
                  CosineAnnealingLR, CyclicLR, OneCycleLR, \
                  CosineAnnealingWarmRestarts

def main_option_menu(menu_list):
    with st.sidebar:
        choice = option_menu('Learning Rate Scheduler', 
                             menu_list, 
                             icons=['house'],
                             menu_icon='app-indicator',
                             default_index=0,
                             styles={
            "container": {"padding": "5!important", 
                          "background-color": "#fafafa"},
            "icon": {"color": "orange", 
                     "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", 
                         "text-align": "left", 
                         "margin":"0px", 
                         "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#02ab21"},
        })
    return choice

def main():
    menu_list = ['About', 'LambdaLR', 'MultiplicativeLR', 'StepLR', 
                 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR',
                 'CyclicLR', 'OneCycleLR', 'CosineAnnealingWarmRestarts']
    # main menu
    choice = main_option_menu(menu_list)
    # define model
    max_epoch = 1000
    max_lr    = 100.
    st.sidebar.write('Basic Options')
    epoch=st.sidebar.slider('Epochs', 0, max_epoch, 100, 10)
    learningRate=st.sidebar.number_input('Learning Rate', 0., max_lr, step=1e-3, value=0.01, format="%.3f")
    basic = BasicModel(learningRate)
    # define container
    container1 = st.container()
    container2 = st.container()

    if choice == 'About':
        col1, col2 = st.columns( [0.8, 0.2] )
        with col1:
            st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
            </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">Guide to Pytorch Learning Rate Scheduler</p>', unsafe_allow_html=True) 
        st.write("This is the app to learn about Pytorch Learning Rate Scheduler and see how it changes according to the setting values.")
        st.write("Reference1 : https://pytorch.org/docs/stable/optim.html (Pytorch Official)")
        st.write("Reference2 : https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook")
    elif choice == 'LambdaLR':   
        # options
        container2.write(f'{choice} Options')
        factor=container2.slider('factor for lr_lambda(lambda epoch : factor ** epoch)', 0., 1., 0.5, 0.01)
        container2.caption(LambdaLR().get_description('lr_lambda'))
        # learn and plot
        scheduler = basic.get_scheduler(lrs.LambdaLR, 
                                        lr_lambda = lambda epoch: factor ** epoch)
        optimizer = basic.get_optim()
        lr = []
        for i in range(epoch):
            optimizer.step()
            lr.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        fig = plot_lr(lr, epoch)

        container1.title(f'{choice}')
        container1.write(LambdaLR().get_description('main'))
        container1.pyplot(fig)
    elif choice == 'MultiplicativeLR':
        # options
        container2.write(f'{choice} Options')
        factor=container2.slider('factor for lr_lambda(lambda epoch : factor ** epoch)', 0., 1., 0.5, 0.01)
        container2.caption(MultiplicativeLR().get_description('lr_lambda'))
        # learn and plot
        scheduler = basic.get_scheduler(lrs.MultiplicativeLR, 
                                        lr_lambda = lambda epoch: factor ** epoch)
        optimizer = basic.get_optim()
        lr = []
        for i in range(epoch):
            optimizer.step()
            lr.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        fig = plot_lr(lr, epoch)
        container1.title(f'{choice}')
        container1.write(MultiplicativeLR().get_description('main'))
        container1.pyplot(fig)
    elif choice == 'StepLR':
        # options
        container2.write(f'{choice} Options')
        step_size=container2.slider('step_size', 1, epoch, 1, 1)
        container2.caption(StepLR().get_description('step_size'))
        gamma=container2.slider('gamma', 0., 1., 0.1, 0.1)
        container2.caption(StepLR().get_description('gamma'))
        # learn and plot
        scheduler = basic.get_scheduler(lrs.StepLR, 
                                        step_size = step_size,
                                        gamma = gamma)
        optimizer = basic.get_optim()
        lr = []
        for i in range(epoch):
            optimizer.step()
            lr.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        fig = plot_lr(lr, epoch)
        container1.title(f'{choice}')
        container1.write(StepLR().get_description('main'))
        container1.pyplot(fig)
    elif choice == 'MultiStepLR':
        # options
        container2.write(f'{choice} Options')
        milestones=container2.multiselect('milestones', 
                                          [ i for i in range(1, epoch)],
                                          help='Must be increasing',
                                          default=[1])
        container2.caption(MultiStepLR().get_description('milestones'))
        container2.caption(sorted(milestones))
        gamma=container2.slider('gamma', 0., 1., 0.1, 0.1)
        container2.caption(MultiStepLR().get_description('gamma'))
        # learn and plot
        scheduler = basic.get_scheduler(lrs.MultiStepLR, 
                                        milestones = sorted(milestones),
                                        gamma = gamma)
        optimizer = basic.get_optim()
        lr = []
        for i in range(epoch):
            optimizer.step()
            lr.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        fig = plot_lr(lr, epoch)
        container1.title(f'{choice}')
        container1.write(MultiStepLR().get_description('main'))
        container1.pyplot(fig)
    elif choice == 'ExponentialLR':
        # options
        container2.write(f'{choice} Options')
        gamma=container2.slider('gamma', 0., 1., 0.1, 0.1)
        container2.caption(ExponentialLR().get_description('gamma'))
        # learn and plot
        scheduler = basic.get_scheduler(lrs.ExponentialLR, 
                                        gamma = gamma)
        optimizer = basic.get_optim()
        lr = []
        for i in range(epoch):
            optimizer.step()
            lr.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        fig = plot_lr(lr, epoch)
        container1.title(f'{choice}')
        container1.write(ExponentialLR().get_description('main'))
        container1.pyplot(fig)
    elif choice == 'CosineAnnealingLR':
        # options
        container2.write(f'{choice} Options')
        T_max=container2.slider('T_max', 1, epoch, 1, 1)
        container2.caption(CosineAnnealingLR().get_description('T_max'))
        eta_min=container2.number_input('eta_min', 0., learningRate, step=1e-3, value=0., format="%.3f")
        container2.caption(CosineAnnealingLR().get_description('eta_min'))
        # learn and plot
        scheduler = basic.get_scheduler(lrs.CosineAnnealingLR, 
                                        T_max = T_max,
                                        eta_min = eta_min)
        optimizer = basic.get_optim()
        lr = []
        for i in range(epoch):
            optimizer.step()
            lr.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        fig = plot_lr(lr, epoch)
        container1.title(f'{choice}')
        container1.write(CosineAnnealingLR().get_description('main'))
        container1.pyplot(fig)
    elif choice == 'CyclicLR':
        # options
        container2.write(f'{choice} Options')
        base_lr=container2.number_input('base_lr', 0., None, value=1e-3, step=1e-3, format="%.3f")
        container2.caption(CyclicLR().get_description('base_lr'))
        max_lr=container2.number_input('max_lr', 0., None, value=0.1, step=1e-3, format="%.3f")
        container2.caption(CyclicLR().get_description('max_lr'))
        step_size_up=container2.number_input('step_size_up', 0, None, step=1, value=2000)
        container2.caption(CyclicLR().get_description('step_size_up'))
        switch_step_size_down=container2.selectbox('step_size_down(On/Off)', ('False','True'), index=0)
        if switch_step_size_down == 'False':
            step_size_down = None
        else:
            step_size_down=container2.number_input('step_size_down', 0, None, step=1,value=10)
        container2.caption(CyclicLR().get_description('step_size_down'))
        mode=container2.selectbox('mode', ('triangular', 'triangular2', 'exp_range'), index=0)
        container2.caption(CyclicLR().get_description('mode'))
        gamma=container2.slider('gamma', 0., 1., 1., 0.1)
        container2.caption(CyclicLR().get_description('gamma'))
        scale_fn=container2.selectbox('scale_fn (Not operated)', ('None',), index=0)
        container2.caption(CyclicLR().get_description('scale_fn'))
        scale_mode=container2.selectbox('scale_mode (Not operated)', ('cycle', 'iterations'), index=0)
        container2.caption(CyclicLR().get_description('scale_mode'))
        cycle_momentum=container2.selectbox('cycle_momentum', ('True', 'False'), index=0)
        if cycle_momentum == "True":
            cycle_momentum = True
        else:
            cycle_momentum = False
        container2.caption(CyclicLR().get_description('cycle_momentum'))
        base_momentum=container2.slider('base_momentum', 0., 1., 0.8, 0.01)
        container2.caption(CyclicLR().get_description('base_momentum'))
        max_momentum=container2.slider('max_momentum', 0., 1., 0.9, 0.01)
        container2.caption(CyclicLR().get_description('max_momentum'))
        # learn and plot
        scheduler = basic.get_scheduler(lrs.CyclicLR, 
                                        base_lr = base_lr,
                                        max_lr  = max_lr,
                                        step_size_up = step_size_up,
                                        step_size_down=step_size_down,
                                        mode = mode,
                                        gamma = gamma,
                                        scale_fn = None,
                                        scale_mode = scale_mode,
                                        cycle_momentum = cycle_momentum,
                                        base_momentum = base_momentum,
                                        max_momentum = max_momentum)
        optimizer = basic.get_optim()
        lr = []
        for i in range(epoch):
            optimizer.step()
            lr.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        fig = plot_lr(lr, epoch)
        container1.title(f'{choice}')
        container1.write(CyclicLR().get_description('main'))
        container1.pyplot(fig)
    elif choice == 'OneCycleLR':
        # options
        container2.write(f'{choice} Options')
        max_lr=container2.number_input('max_lr', 0., max_lr, value=0.1, step=1e-3, format="%.3f")
        container2.caption(OneCycleLR().get_description('max_lr'))
        switch_step=container2.selectbox('select steps or epochs', ('total_steps','epochs'), index=1)
        if switch_step == "total_steps":
            total_steps=container2.number_input('total_steps', 0, None, step=1, value=epoch)
            container2.caption(OneCycleLR().get_description('total_steps'))
            epochs = None
            steps_per_epoch=None
        else:
            total_steps= None
            epochs=container2.number_input('epochs', 0, None, step=1, value=10)
            container2.caption(OneCycleLR().get_description('epochs'))
            steps_per_epoch=container2.number_input('steps_per_epoch', 0, None, step=1, value=10)
            container2.caption(OneCycleLR().get_description('steps_per_epoch'))
        pct_start=container2.slider('pct_start', 0., 1., 0.3, 0.01)
        container2.caption(OneCycleLR().get_description('pct_start'))
        anneal_strategy=container2.selectbox('anneal_strategy', ('cos', 'linear'), index=0)
        container2.caption(OneCycleLR().get_description('anneal_strategy'))
        cycle_momentum=container2.selectbox('cycle_momentum', ('True', 'False'), index=0)
        if cycle_momentum == "True":
            cycle_momentum = True
        else:
            cycle_momentum = False
        container2.caption(OneCycleLR().get_description('cycle_momentum'))
        base_momentum=container2.slider('base_momentum', 0., 1., 0.85, 0.01)
        container2.caption(OneCycleLR().get_description('base_momentum'))
        max_momentum=container2.slider('max_momentum', 0., 1., 0.95, 0.01)
        container2.caption(OneCycleLR().get_description('max_momentum'))
        div_factor=container2.number_input('div_factor', 0., None, step=0.1, value=25.)
        container2.caption(OneCycleLR().get_description('div_factor'))
        final_div_factor=container2.number_input('final_div_factor', 0., None, value=1e+4, step=0.1)
        container2.caption(OneCycleLR().get_description('final_div_factor'))
        three_phase=container2.selectbox('three_phase', ('True', 'False'), index=1)
        if cycle_momentum == "True":
            three_phase = True
        else:
            three_phase = False
        # learn and plot
        scheduler = basic.get_scheduler(lrs.OneCycleLR, 
                                        max_lr  = max_lr,
                                        total_steps = total_steps,
                                        epochs = epochs,
                                        steps_per_epoch = steps_per_epoch,
                                        pct_start = pct_start,
                                        anneal_strategy = anneal_strategy,
                                        cycle_momentum = cycle_momentum,
                                        base_momentum = base_momentum,
                                        max_momentum = max_momentum,
                                        div_factor = div_factor,
                                        final_div_factor = final_div_factor,
                                        three_phase = three_phase)
        optimizer = basic.get_optim()
        lr = []
        for i in range(epoch):
            optimizer.step()
            lr.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        fig = plot_lr(lr, epoch)
        container1.title(f'{choice}')
        container1.write(OneCycleLR().get_description('main'))
        container1.pyplot(fig)
    elif choice == "CosineAnnealingWarmRestarts":
        # options
        container2.write(f'{choice} Options')
        T_0=container2.slider('T_0', 0, epoch, 10, 1)
        container2.caption(CosineAnnealingWarmRestarts().get_description('T_0'))
        T_mult=container2.slider('T_mult', 0, epoch, 1, 1)
        container2.caption(CosineAnnealingWarmRestarts().get_description('T_mult'))
        eta_min=container2.slider('eta_min', 0., learningRate, 0., 0.001)
        container2.caption(CosineAnnealingWarmRestarts().get_description('eta_min'))
        # learn and plot
        scheduler = basic.get_scheduler(lrs.CosineAnnealingWarmRestarts, 
                                        T_0 = T_0,
                                        T_mult = T_mult,
                                        eta_min = eta_min)
        optimizer = basic.get_optim()
        lr = []
        for i in range(epoch):
            optimizer.step()
            lr.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        fig = plot_lr(lr, epoch)
        container1.title(f'{choice}')
        container1.write(CosineAnnealingWarmRestarts().get_description('main'))
        container1.pyplot(fig)
        
if __name__ == "__main__":
    main()
