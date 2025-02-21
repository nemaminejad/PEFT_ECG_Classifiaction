
import matplotlib.pyplot as plt
import numpy as np
import gc

def plot_ecg_signals(arr,idx,save_path):
  """Plots 12 ECG signals with normalization.

  Args:
    arr (np.ndarray):  of shape (5000, 12) representing 12 ECG leads.
    idx (int)
    save_path (str) : path/to/image/folder
  """

  num_leads = arr.shape[2]
  fig, axes = plt.subplots(num_leads, 1, figsize=(10, 2 * num_leads))
  # print('num leads: ', num_leads)
  for i in range(num_leads):
    # Normalize the signal to the range [-1, 1]
    signal = (arr[0,:, i] - np.min(arr[0,:, i])) / (np.max(arr[0,:, i]) - np.min(arr[0,:, i])) * 2 - 1
    # print(signal.shape)
    axes[i].plot(signal)

    # Remove axis numbers and frame
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['bottom'].set_visible(False)
    axes[i].spines['left'].set_visible(False)
    del signal

  plt.tight_layout()
  plt.savefig(save_path+f'/{idx}_plot.png', dpi=100)
  plt.close(fig)
  del arr
  
if __name__ =='__main__':
    
  path = './data/PTB-XL/physionet.org/files/ptb-xl/1.0.3/'
  
  for fold in ['train','test','val']
    
    for label in ['0','1','2','3','4']:
      input_path = os.path.join(path,'PTB-XL-npy',fold,label)
      out_path = os.path.join(path,'PTB-XL-img',fold,label)
      # create output folder
      os.makedirs(out_path, exist_ok = True)
      
      for np_case in os.listdir(input_path):
        if '.npy' in np_case:
          curr_data = np.load(os.path.join(input_path,np_case))
          idx = np_case.split('.npy')[0]
          if not os.path.exists(out_path+f'/{idx}_plot.png'):
            plot_ecg_signals(curr_data, idx,out_path) # plots the first ECG signal in test_data
            del curr_data
            gc.collect() # triggers the garbage collection process to reclaim memory by clearing unused or unreachable objects that are no longer needed