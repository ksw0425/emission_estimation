import os
import glob
import numpy as np
import pandas as pd

def hampel_filter_series(x, window_size=7, n_sigmas=3):
    new_x = x.copy()
    rolling_median = x.rolling(window_size, center=True).median()
    mad = lambda s: np.median(np.abs(s - np.median(s)))
    rolling_mad = x.rolling(window_size, center=True).apply(mad)
    threshold = n_sigmas * 1.4826 * rolling_mad
    outliers = (x - rolling_median).abs() > threshold
    new_x[outliers] = rolling_median[outliers]
    return new_x

def load_and_preprocess_speed(input_dir,
                              fps=15,
                              hampel_win=7,
                              hampel_sig=3,
                              ma_window=5,
                              k_frames=3):
    # 1) .npy → DataFrame
    records = []
    for npy_path in glob.glob(os.path.join(input_dir, '*.npy')):
        frame = int(os.path.splitext(os.path.basename(npy_path))[0])
        data = np.load(npy_path, allow_pickle=True)
        # 1D/2D 모두 처리
        entries = data if data.ndim == 1 else data[:, :3]
        for obj_id, speed, cls in entries:
            records.append({
                'frame': frame,
                'vehicle_id': int(obj_id),
                'speed_m_s': float(speed),
                'ClassID': int(cls)
            })
    df = pd.DataFrame(records)

    out = []
    for vid, g in df.sort_values(['vehicle_id','frame']).groupby('vehicle_id'):
        # 2) 누락 프레임 채우기
        orig = g.set_index('frame')
        all_frames = np.arange(orig.index.min(), orig.index.max() + 1)
        g2 = orig.reindex(all_frames)

        # 3) Hampel → 이동평균 → 보간
        g2['speed_m_s'] = hampel_filter_series(
            g2['speed_m_s'], window_size=hampel_win, n_sigmas=hampel_sig)
        g2['speed_m_s'] = g2['speed_m_s'].rolling(
            window=ma_window, center=True, min_periods=1).mean()
        g2['speed_m_s'] = g2['speed_m_s'].interpolate(method='linear')

        # 4) ClassID 보간 후 최빈값 통일
        g2['ClassID'] = g2['ClassID'].ffill().bfill().astype(int)
        mode_cls = g2['ClassID'].mode().iloc[0]
        g2['ClassID'] = mode_cls

        # 5) k-프레임 차분으로 가속도 계산
        g2['speed_prev'] = g2['speed_m_s'].shift(k_frames)
        # Δt = k_frames / fps
        dt = k_frames / fps
        g2['Accel_m_s2'] = ((g2['speed_m_s'] - g2['speed_prev']) / dt).fillna(0)

        # 6) 최종 컬럼 정리
        g2 = g2.reset_index().rename(columns={'index': 'frame'})
        g2['Time'] = g2['frame'] / fps
        g2['vehicle_id'] = vid
        out.append(g2[['frame','vehicle_id','Time','speed_m_s','Accel_m_s2','ClassID']])

    return pd.concat(out, ignore_index=True)

def to_pyemission_format(df_proc):
    return df_proc.rename(columns={
        'frame': 'frame',
        'vehicle_id': 'VehicleID',
        'Time':       'Time',
        'speed_m_s':  'Speed_m_s',
        'Accel_m_s2': 'Accel_m_s2',
        'ClassID':    'ClassID'
    })[['frame','VehicleID','Time','Speed_m_s','Accel_m_s2','ClassID']]


# ── 사용 예시 ──
input_npy_dir = '/content/drive/MyDrive/Colab Notebooks/Main/Track/output/speed'     # .npy 파일 경로
fps = 15

# 전처리 + PyEmission 포맷 변환
df_ready    = load_and_preprocess_speed(input_npy_dir, fps=fps)
df_emission = to_pyemission_format(df_ready)

# CSV로 저장
output_csv = '/content/drive/MyDrive/Colab Notebooks/Main/Track/output/speed_preprocessed.csv'
df_emission.to_csv(output_csv, index=False)

print(f"✅ 전처리 완료, CSV 저장 경로: {output_csv}")
