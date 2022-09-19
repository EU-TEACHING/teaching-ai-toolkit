import seaborn as sns
from matplotlib import pyplot as plt


def _visualize_loss(self):
    plt.plot(self.model_history.history["loss"], label="Training Loss")
    plt.plot(self.model_history.history["val_loss"], label="Test Loss")
    plt.legend()
    plt.show()


def visualize_anomalies(self, anomalous_data_indices):
    df_subset = self.anomaly.iloc[anomalous_data_indices]
    cols = ['ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
            'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
            'ct_srv_dst']
    fig, axes = plt.subplots(nrows=11, ncols=1)
    for i, col in enumerate(cols):
        g = sns.lineplot(data=self.anomaly, x=self.anomaly.index, y=col, ax=axes[i])

        line_position = df_subset.index.to_list()
        for pos in line_position:
            axes[i].axvline(x=pos, color='r', linestyle=':')

    plt.show()
