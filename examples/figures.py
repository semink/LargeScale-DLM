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
    fwy_680_milepost = sensors_in_fwy_680['Abs_PM'].diff().abs().fillna(0).cumsum()
    fwy_milepost = pd.concat([fwy_680_milepost, fwy_680_milepost.values[-1] + sensors_in_fwy_280['Abs_PM']])
    print(f'Number of sensors: {sensors_in_fwy.shape[0]}')
    df_raw.columns = df_raw.columns.astype(df_meta.index.dtype)
    df_fwy = df_raw[sensors_in_fwy.index]

    day = '2017-05-22'
    title = df_fwy[day].index[0].strftime('%Y-%m-%d [%a.]')
    x_labels = df_fwy[day].index.strftime('%H').astype(int)
    plt.pcolor(df_fwy[day].T, cmap='Greys_r', vmin=10, vmax=80)
    selected_sensors = {'A': 407200, 'B': 407206, 'C': 401403}
    yticks_sensors = [sensors_in_fwy.index.get_loc(selected_sensors[sensor])-0.5 for sensor in selected_sensors]
    yticks = np.arange(len(sensors_in_fwy.index))-0.5
    ytick_limit_idx = [2, int(len(yticks)*2/5)]
    yticks_figure = [yticks[ytick_limit_idx[0]], *yticks_sensors, yticks[ytick_limit_idx[1]]]
    ylabel_figure = [0,
                     *[f'{fwy_milepost.loc[selected_sensors[sensor]]:.1f}\n({sensor})' for sensor in selected_sensors],
                     f'{fwy_milepost.values[ytick_limit_idx[1]]:.1f}']
    plt.yticks(yticks_figure, ylabel_figure)
    xticks = np.arange(len(df_fwy[day].index))
    plt.xticks(xticks, x_labels)
    plt.locator_params(axis='x', nbins=10)
    plt.xlabel('Time [hour]')
    plt.ylabel('Milepost [mile]')
    plt.xlim([xticks[int(len(xticks)/2)], xticks[-1]])
    plt.ylim([yticks[ytick_limit_idx[0]], yticks[ytick_limit_idx[1]]])
    clb =plt.colorbar()
    clb.set_label('Speed [mph]')
    # plt.show()
    plt.savefig('fig_1_c.pdf')
    print('done.')


if __name__ == '__main__':
    main()
