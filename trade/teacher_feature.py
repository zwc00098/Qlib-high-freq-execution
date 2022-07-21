import pandas as pd 
import os

data_path = 'C:/Users/16053/qlib-high-freq-execution/examples/data'
feature_path = os.path.join(data_path, 'feature/teacher/')
if not os.path.exists(feature_path):
    os.makedirs(feature_path)

os.environ["OUTPUT_DIR"] = 'C:/Users/16053/qlib-high-freq-execution/examples/trade/log'
log_file_1 = os.path.join(os.environ.get('OUTPUT_DIR'),'example/OPDT_b_1/test/')
log_file_2 = os.path.join(os.environ.get('OUTPUT_DIR'),'example/OPDT_b_2/test/')
log_file_3 = os.path.join(os.environ.get('OUTPUT_DIR'),'example/OPDT_b_3/test/')

files = os.listdir(log_file_1)

for f in files:
    if f.endswith(".log"):
        df1 = pd.read_pickle(log_file_1 + f)
        df2 = pd.read_pickle(log_file_2 + f)
        df3 = pd.read_pickle(log_file_3 + f)

        #df['datetime'] = df.index.get_level_values(1).map(lambda x: x[1])
        df1['datetime'] = df1.index.get_level_values(1)
        df1.set_index('datetime', append=True, drop=True, inplace=True)
        action_1 = df1['action']
        action_1 = action_1.reset_index(level=1, drop=True)
        action_1.index = action_1.index.map(lambda x: (x[0], x[1], x[2].time()))
        action_1 = action_1.unstack().iloc[:, ::30] * 2
        action_1 = action_1.fillna(0)
        train_action_1 = action_1.astype("int")
        final_1 = train_action_1
        
        df2['datetime'] = df2.index.get_level_values(1)
        df2.set_index('datetime', append=True, drop=True, inplace=True)
        action_2 = df2['action']
        action_2 = action_2.reset_index(level=1, drop=True)
        action_2.index = action_2.index.map(lambda x: (x[0], x[1], x[2].time()))
        action_2 = action_2.unstack().iloc[:, ::30] * 2
        action_2 = action_2.fillna(0)
        train_action_2 = action_2.astype("int")
        final_2 = train_action_2
        
        df3['datetime'] = df3.index.get_level_values(1)
        df3.set_index('datetime', append=True, drop=True, inplace=True)
        action_3 = df3['action']
        action_3 = action_3.reset_index(level=1, drop=True)
        action_3.index = action_3.index.map(lambda x: (x[0], x[1], x[2].time()))
        action_3 = action_3.unstack().iloc[:, ::30] * 2
        action_3 = action_3.fillna(0)
        train_action_3 = action_3.astype("int")
        final_3 = train_action_3
        
        final = 0.5 * final_1 + 0.4 * final_2 + 0.1 * final_3
        
        final.to_pickle(feature_path + f[:-4] + '.pkl')