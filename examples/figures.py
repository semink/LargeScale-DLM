from lsdlm import utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def main():
    print('loading dataset...', end=' ')
    df_raw, df_meta = utils.load_dataset()
    sensors_in_fwy_680=df_meta[(df_meta.Fwy == 680) & (df_meta.Dir == 'S')]
    sensors_in_fwy_280=df_meta[(df_meta.Fwy == 280) & (df_meta.Dir == 'N')]
    sensors_in_fwy = pd.concat([sensors_in_fwy_680,sensors_in_fwy_280])
    print(f'Number of sensors: {sensors_in_fwy.shape[0]}')
    df_raw.columns = df_raw.columns.astype(df_meta.index.dtype)
    df_fwy = df_raw[sensors_in_fwy.index]

    day = '2017-05-22'
    title = df_fwy[day].index[0].strftime('%Y-%m-%d [%a.]')
    x_labels = df_fwy[day].index.strftime('%H').astype(int)
    plt.pcolor(df_fwy[day].T, cmap='Greys_r', vmin=10, vmax=80)
    selected_sensors = {'A': 407200, 'B': 407206, 'C': 401403}
    plt.yticks([sensors_in_fwy.index.get_loc(selected_sensors[sensor])-0.5 for sensor in selected_sensors], [sensor for sensor in selected_sensors])
    plt.xticks(np.arange(len(df_fwy[day].index)),
               x_labels)
    plt.locator_params(axis='x', nbins=10)
    plt.xlabel('Time [hour]')
    clb =plt.colorbar()
    clb.set_label('Speed [mph]')
    plt.savefig('fig_1_c.pdf')
    print('done.')


if __name__ == '__main__':
    main()
