import pandas as pd
from bokeh.plotting import figure, show, output_file
from datetime import datetime

df = pd.read_csv('smile_records.csv')
smile_ratios = list(df['smile_ratio'])
sm = [round(s, 3) for s in smile_ratios]
times = list(df['times'])
date_time=[datetime.strptime(d,'%Y-%m-%d %H:%M:%S.%f') for d in times]
p = figure(plot_width=800, plot_height=400, x_axis_type='datetime')
p.line(date_time, sm, alpha=0.5)
show(p)
output_file('graph.html')
