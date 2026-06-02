import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoLocator
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.transforms as transforms
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

class RiskModelVisualizer(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        style = ttk.Style(self)
        style.configure('.', font=('Arial', 10))
        
        self.title("Risk Model Analyzer")
        self.geometry("1600x900")

        self.df = None
        self.ani_playback = None
        self.df_filepath = None
        self.animation_paused = False
        
        self.fig_width = 16
        self.fig_height = 8 
        self.tick_fontsize = 16
        self.linewidth = 4
        self.traj_linewidth = 3
        self.show_legend = True

        self.car_length, self.car_width = 4.74, 2.02
        self.Q_veh, self.Q_ped, self.epsilon_R, self.delta_d = 2.0, 1.0, 1.0, 0.1
        self.color_CRV, self.color_ttc, self.color_DSF = '#D9534F', '#0275D8', '#5CB85C'
        self.color_traj, self.color_vehicle_fill, self.color_ped_fill = 'black', '#808080', '#9400D3'
        self.style_CRV, self.style_ttc, self.style_DSF = '-', ':', '--'
        self.style_car_traj, self.style_ped_traj = '-', '--'

        self.CRVate_widgets()

    def CRVate_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=0)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        control_frame = ttk.LabelFrame(left_panel, text="Controls")
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        btn_load = ttk.Button(control_frame, text="Load CSV File", command=self.load_file)
        btn_load.pack(padx=5, pady=5, fill=tk.X)
        
        btn_replay = ttk.Button(control_frame, text="Replay Animation", command=self.run_analysis)
        btn_replay.pack(padx=5, pady=5, fill=tk.X)
        
        self.pause_button = ttk.Button(control_frame, text="Pause Animation", command=self.toggle_animation_pause, state=tk.DISABLED)
        self.pause_button.pack(padx=5, pady=5, fill=tk.X)
        
        btn_save_img = ttk.Button(control_frame, text="Save Current Image", command=self.save_figure)
        btn_save_img.pack(padx=5, pady=5, fill=tk.X)
        
        btn_save_vid = ttk.Button(control_frame, text="Save Animation Video", command=self.save_animation)
        btn_save_vid.pack(padx=5, pady=5, fill=tk.X)
        
        self.status_label = ttk.Label(control_frame, text="Status: Waiting for data file...", wraplength=200)
        self.status_label.pack(padx=5, pady=5, fill=tk.X)
        
        results_frame = ttk.LabelFrame(left_panel, text="Quantitative Results")
        results_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        self.results_label = ttk.Label(results_frame, text="Results will be shown after loading a file.", wraplength=250)
        self.results_label.pack(side=tk.LEFT, padx=5, pady=5)

        canvas_container = ttk.Frame(main_frame)
        canvas_container.grid(row=0, column=1, sticky="nsew")
        self.CRVate_matplotlib_canvas(canvas_container)

    def CRVate_matplotlib_canvas(self, parent):
        self.fig = plt.Figure(figsize=(self.fig_width, self.fig_height), dpi=100)
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = self.tick_fontsize
        
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25)
        
        self.ax_anim = self.fig.add_subplot(gs[0])
        self.ax_graph = self.fig.add_subplot(gs[1])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def toggle_animation_pause(self):
        if not hasattr(self, 'ani_playback') or self.ani_playback is None or not self.ani_playback.event_source:
            return
        
        if self.animation_paused:
            self.ani_playback.resume()
            self.pause_button.config(text="Pause Animation")
            self.status_label.config(text="Animation playing...")
        else:
            self.ani_playback.pause()
            self.pause_button.config(text="Resume Animation")
            self.status_label.config(text="Animation paused.")
        self.animation_paused = not self.animation_paused

    def load_file(self):
        filepath = filedialog.askopenfilename(title="Select Vehicle-Pedestrian Interaction Data File", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not filepath: return
        try:
            expected_columns = [
                't', 'car_x', 'car_y', 'car_z', 'speed', 'acc', 
                'ped_x', 'ped_y', 'ped_z'
            ]
            self.df_original = pd.read_csv(filepath, header=None)
            
            if len(self.df_original.columns) != len(expected_columns):
                messagebox.showerror("File Error", f"Invalid file format. Expected {len(expected_columns)} columns, but found {len(self.df_original.columns)}.")
                return
                
            self.df_original.columns = expected_columns
            self.df_filepath = filepath
            self.status_label.config(text=f"Loaded: {os.path.basename(filepath)}")
            self.run_analysis()
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to load or process the file:\n{e}")

    def save_figure(self):
        if self.df is None or not hasattr(self, 'fig'): messagebox.showinfo("Info", "Please load data and generate a chart first."); return
        default_name = os.path.splitext(os.path.basename(self.df_filepath or "analysis"))[0] + "_capture.png"
        filepath = filedialog.asksaveasfilename(title="Save Current Image", initialfile=default_name, defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("PDF Vector File", "*.pdf")])
        if not filepath: return
        try:
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.1)
            messagebox.showinfo("Success", f"Chart successfully saved to:\n{filepath}")
        except Exception as e: messagebox.showerror("Save Failed", f"An error occurred while saving the chart:\n{e}")

    def save_animation(self):
        if self.df is None: messagebox.showinfo("Info", "Please load data to generate an animation."); return
        default_name = os.path.splitext(os.path.basename(self.df_filepath or "analysis"))[0] + "_animation.mp4"
        filepath = filedialog.asksaveasfilename(title="Save Animation as MP4 Video", initialfile=default_name, defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
        if not filepath: return
        try:
            self.setup_figure_for_animation()
            ani_for_save = animation.FuncAnimation(self.fig, self.animate, frames=len(self.df), blit=False, repeat=False)
            writer = animation.writers['ffmpeg'](fps=20, metadata=dict(artist='RiskModelVisualizer'), bitrate=1800)
            self.status_label.config(text="Saving video, please wait..."); self.update_idletasks()
            ani_for_save.save(filepath, writer=writer, dpi=200)
            messagebox.showinfo("Success", f"Animation successfully saved to:\n{filepath}")
            self.status_label.config(text=f"Video saved: {os.path.basename(filepath)}")
        except FileNotFoundError:
            messagebox.showerror("Error", "Save failed: FFmpeg not found.\nPlease install FFmpeg and ensure its path is in your system's environment variables.")
            self.status_label.config(text="Status: FFmpeg not found, save failed.")
        except Exception as e: messagebox.showerror("Save Failed", f"An unknown error occurred while saving the video:\n{e}"); self.status_label.config(text="Status: Video save failed.")

    def run_analysis(self):
        if not hasattr(self, 'df_original') or self.df_original is None: messagebox.showinfo("Info", "Please load a CSV data file first."); return
        self.status_label.config(text="Analyzing data and preparing animation..."); self.update_idletasks()
        if hasattr(self, 'pause_button'): self.pause_button.config(state=tk.NORMAL)
        self.df = self.df_original.copy()
        self.prepare_simulation_data()
        self.display_quantitative_results()
        self.setup_animation_and_run()

    def display_quantitative_results(self):
        min_ttc_idx = np.nanargmin(self.ttc_values) if ~np.all(np.isnan(self.ttc_values)) else -1
        self.min_ttc = {'value': self.ttc_values[min_ttc_idx] if min_ttc_idx != -1 else 0,
                        'time': self.df['t'].iloc[min_ttc_idx] if min_ttc_idx != -1 else -1}

        min_CRV_idx = np.nanargmin(self.CRV_magnitudes) if ~np.all(np.isnan(self.CRV_magnitudes)) else -1
        self.min_CRV_mag = {'value': self.CRV_magnitudes[min_CRV_idx] if min_CRV_idx != -1 else 0,
                            'time': self.df['t'].iloc[min_CRV_idx] if min_CRV_idx != -1 else -1}

        max_dsf_idx = np.nanargmax(self.DSF_values) if ~np.all(np.isnan(self.DSF_values)) else -1
        self.max_DSF = {'value': self.DSF_values[max_dsf_idx] if max_dsf_idx != -1 else 0,
                        'time': self.df['t'].iloc[max_dsf_idx] if max_dsf_idx != -1 else -1}
        
        min_ttc_val = self.min_ttc['value']; min_ttc_time = self.min_ttc['time']
        min_CRV_mag_val = self.min_CRV_mag['value']; min_CRV_mag_time = self.min_CRV_mag['time']
        max_DSF_val = self.max_DSF['value']; max_DSF_time = self.max_DSF['time']
        
        results_text = (f"Min TTC: {min_ttc_val:.3f}s (at {min_ttc_time:.2f}s)\n"
                        f"Min CRV: {min_CRV_mag_val:.3f}s (at {min_CRV_mag_time:.2f}s)\n"
                        f"Max DSF: {max_DSF_val:.3f} (at {max_DSF_time:.2f}s)")
        self.results_label.config(text=results_text)
    
    def prepare_simulation_data(self):
        if self.df is None: return
        if 't' in self.df.columns: self.df = self.df.sort_values(by='t', ascending=True).reset_index(drop=True)
        
        self.CRV_magnitudes, self.CRV_vectors, self.DSF_values, self.DSF_plot_values = [], [], [], []
        
        dt = self.df['t'].diff().fillna(method='bfill').fillna(method='ffill')

        self.df['car_vx'] = self.df['car_x'].diff().fillna(0) / dt
        self.df['car_vy'] = self.df['car_y'].diff().fillna(0) / dt
        self.df['ped_vx'] = self.df['ped_x'].diff().fillna(0) / dt
        self.df['ped_vy'] = self.df['ped_y'].diff().fillna(0) / dt

        window_size = 7
        self.df['car_vx_smooth'] = self.df['car_vx'].rolling(window=window_size, min_periods=1, center=True).mean()
        self.df['car_vy_smooth'] = self.df['car_vy'].rolling(window=window_size, min_periods=1, center=True).mean()
        self.df['car_vx_smooth'].fillna(method='bfill', inplace=True); self.df['car_vx_smooth'].fillna(method='ffill', inplace=True)
        self.df['car_vy_smooth'].fillna(method='bfill', inplace=True); self.df['car_vy_smooth'].fillna(method='ffill', inplace=True)
        
        total_displacement_y = self.df['car_y'].iloc[-1] - self.df['car_y'].iloc[0]
        car_moving_forward = total_displacement_y > 0

        if car_moving_forward:
            car_front_y = self.df['car_y'] + self.car_length / 2
            signed_longitudinal_distance = car_front_y - self.df['ped_y']
        else:
            car_front_y = self.df['car_y'] - self.car_length / 2
            signed_longitudinal_distance = -(car_front_y - self.df['ped_y'])
            
        self.df['plot_distance'] = signed_longitudinal_distance
        self.df['abs_distance'] = abs(signed_longitudinal_distance)
        self.df['plot_car_x'] = -self.df['car_x']
        self.df['plot_ped_x'] = -self.df['ped_x']

        raw_ttc_values = []
        intersection_x = self.df['car_x'].mean()
        intersection_y = self.df['ped_y'].mean()
        p_int = np.array([intersection_x, intersection_y])
        
        for i in range(len(self.df)):
            ttc = np.nan 
            v_car_vec = np.array([self.df['car_vx'][i], self.df['car_vy'][i]])
            p_ped = np.array([self.df['ped_x'][i], self.df['ped_y'][i]])
            v_ped_vec = np.array([self.df['ped_vx'][i], self.df['ped_vy'][i]])
            if car_moving_forward: car_front_y_i = self.df['car_y'][i] + self.car_length / 2
            else: car_front_y_i = self.df['car_y'][i] - self.car_length / 2
            p_veh = np.array([self.df['car_x'][i], car_front_y_i])
            vec_veh_to_int = p_int - p_veh; dist_veh_to_int = np.linalg.norm(vec_veh_to_int)
            time_to_reach_veh = np.inf
            if dist_veh_to_int > 1e-6:
                v_veh_proj = np.dot(v_car_vec, vec_veh_to_int) / dist_veh_to_int
                if v_veh_proj > 0.1: time_to_reach_veh = dist_veh_to_int / v_veh_proj
            vec_ped_to_int = p_int - p_ped; dist_ped_to_int = np.linalg.norm(vec_ped_to_int)
            time_to_reach_ped = np.inf
            if dist_ped_to_int > 1e-6:
                v_ped_proj = np.dot(v_ped_vec, vec_ped_to_int) / dist_ped_to_int
                if v_ped_proj > 0.1: time_to_reach_ped = dist_ped_to_int / v_ped_proj
            if time_to_reach_veh <= time_to_reach_ped:
                ttc = min(time_to_reach_veh, 15.0)
            raw_ttc_values.append(ttc)

            car_pos = np.array([self.df['car_x'][i], self.df['car_y'][i]])
            v_car = np.array([self.df['car_vx'][i], self.df['car_vy'][i]])
            ped_pos = np.array([self.df['ped_x'][i], self.df['ped_y'][i]])
            v_ped = np.array([self.df['ped_vx'][i], self.df['ped_vy'][i]])
            
            grid_points = []
            for p in np.linspace(-self.car_width/2, self.car_width/2, 8): grid_points.append(car_pos + np.array([p, self.car_length/2])); grid_points.append(car_pos + np.array([p, -self.car_length/2]))
            for p in np.linspace(-self.car_length/2, self.car_length/2, 10): grid_points.append(car_pos + np.array([-self.car_width/2, p])); grid_points.append(car_pos + np.array([self.car_width/2, p]))

            total_CRV_vector, count = np.array([0.0, 0.0]), 0
            v_rel_CRV = v_ped - v_car
            for grid_pos in grid_points:
                p_rel = ped_pos - grid_pos; p_rel_mag = np.linalg.norm(p_rel); v_rel_mag = np.linalg.norm(v_rel_CRV)
                if p_rel_mag > 1e-6 and v_rel_mag > 1e-6:
                    cos_theta = np.dot(p_rel, v_rel_CRV) / (p_rel_mag * v_rel_mag)
                    if abs(cos_theta) > 1e-6:
                        pttc_for_CRV = min(abs(p_rel_mag / (v_rel_mag * abs(cos_theta))), 15.0)
                        if pttc_for_CRV < 0.1: pttc_for_CRV = 0.1
                        total_CRV_vector += (pttc_for_CRV * (p_rel / p_rel_mag)); count += 1
            CRV_vector = total_CRV_vector / count if count > 0 else np.array([0., 15.0])
            CRV_magnitude = np.linalg.norm(CRV_vector)
            self.CRV_magnitudes.append(CRV_magnitude)
            self.CRV_vectors.append(CRV_vector)

            v_ij = v_ped - v_car
            v_ij_mag = np.linalg.norm(v_ij)
            dsf_risks_for_frame = []
            
            for grid_pos in grid_points:
                p_ij_grid = ped_pos - grid_pos
                d = np.linalg.norm(p_ij_grid)
                dsf_point_risk = 0.0
                if d > 1e-6:
                    dsf_point_risk = (self.Q_veh * self.Q_ped * v_ij_mag) / (self.epsilon_R * (d + self.delta_d))
                dsf_risks_for_frame.append(dsf_point_risk)
            
            avg_DSF_risk = np.mean(dsf_risks_for_frame) if dsf_risks_for_frame else 0.0
            self.DSF_values.append(avg_DSF_risk)
            
            k = 0.1
            DSF_plot_val = np.exp(-k * avg_DSF_risk)
            self.DSF_plot_values.append(DSF_plot_val)

        self.ttc_values = np.array(raw_ttc_values)
        conflict_indices = np.where(signed_longitudinal_distance <= 0)[0]
        conflict_idx = conflict_indices[0] if len(conflict_indices) > 0 else len(self.df)
        if conflict_idx < len(self.ttc_values):
            self.ttc_values[conflict_idx:] = np.nan

        self.CRV_magnitudes, self.DSF_values = np.array(self.CRV_magnitudes), np.array(self.DSF_values)
        self.DSF_plot_values = np.array(self.DSF_plot_values)
        
        def smooth_data_advanced(data):
            data_clean = data.copy(); valid_mask = ~np.isnan(data_clean)
            if np.sum(valid_mask) > 5:
                valid_data = data_clean[valid_mask]; mean_val, std_val = np.mean(valid_data), np.std(valid_data)
                if std_val > 1e-6:
                    z_scores = np.abs((valid_data - mean_val) / std_val)
                    median_val = np.median(valid_data[z_scores <= 3]) if np.sum(z_scores <= 3) > 0 else mean_val
                    data_clean[valid_mask & (np.abs((data_clean - mean_val) / std_val) > 3)] = median_val
            
            smoothed = pd.Series(data_clean).rolling(window=5, min_periods=1, center=True).mean()
            final_smoothed = smoothed.to_numpy()
            
            valid_mask_smooth = ~np.isnan(final_smoothed)
            if np.sum(valid_mask_smooth) > 3:
                final_smoothed[valid_mask_smooth] = gaussian_filter1d(final_smoothed[valid_mask_smooth], sigma=1.0)
                
            return final_smoothed
        
        self.ttc_smooth = smooth_data_advanced(self.ttc_values)
        self.CRV_smooth = smooth_data_advanced(self.CRV_magnitudes)
        self.DSF_smooth = smooth_data_advanced(self.DSF_plot_values)
    
    def setup_figure_for_animation(self):
        for ax in [self.ax_anim, self.ax_graph]: ax.clear()
        if hasattr(self, 'ax_DSF'): self.ax_DSF.remove()
        
        self.ax_anim.set_title('Vehicle-Pedestrian Interaction', pad=20, fontsize=self.tick_fontsize)
        self.ax_anim.set_aspect('equal'); self.ax_anim.set_xlabel('X Coordinate (m)'); self.ax_anim.set_ylabel('Y Coordinate (m)')
        self.car_rect = patches.Rectangle((-self.car_length / 2, -self.car_width / 2), self.car_length, self.car_width, fc=self.color_vehicle_fill, alpha=0.8, label='Vehicle')
        self.ax_anim.add_patch(self.car_rect)
        self.ped_point, = self.ax_anim.plot([], [], 'o', color=self.color_ped_fill, markersize=8, label='Pedestrian')
        self.car_trail, = self.ax_anim.plot([], [], color=self.color_traj, linestyle=self.style_car_traj, lw=self.traj_linewidth, alpha=0.6, label='Car Path')
        self.ped_trail, = self.ax_anim.plot([], [], color=self.color_traj, linestyle=self.style_ped_traj, lw=self.traj_linewidth, alpha=0.8, label='Ped Path')
        self.CRV_arrow = self.ax_anim.quiver([], [], [], [], color=self.color_CRV, scale=10, units='inches', width=0.04, label='CRV')
        self.CRV_text = self.ax_anim.text(0, 0, '', fontsize=9, color=self.color_CRV, ha='center', va='bottom', weight='bold')
        self.info_text = self.ax_anim.text(0.02, 1.02, '', transform=self.ax_anim.transAxes, fontsize=self.tick_fontsize)
        self.ax_anim.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=self.tick_fontsize-2)
        self.ax_anim.grid(False)
        self.ax_anim.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        self.ax_graph.set_title("Risk Models vs. Longitudinal Distance", fontsize=self.tick_fontsize)
        self.ax_graph.set_xlabel("Longitudinal Distance (m)")
        self.ax_graph.set_ylabel("CRV/TTC Value", color=self.color_CRV)
        self.ax_DSF = self.ax_graph.twinx()
        self.ax_DSF.set_ylabel("DSF (exp-decay)", color=self.color_DSF)
        self.ax_DSF.tick_params(axis='y', labelcolor=self.color_DSF, labelsize=self.tick_fontsize)
        self.ax_graph.tick_params(axis='y', labelcolor=self.color_CRV, labelsize=self.tick_fontsize)
        self.ax_graph.tick_params(axis='x', labelsize=self.tick_fontsize)
        
        self.CRV_line, = self.ax_graph.plot([], [], color=self.color_CRV, linestyle=self.style_CRV, label='CRV (s)', lw=self.linewidth)
        self.ttc_line, = self.ax_graph.plot([], [], color=self.color_ttc, linestyle=self.style_ttc, label='TTC (s)', lw=self.linewidth)
        self.DSF_line, = self.ax_DSF.plot([], [], color=self.color_DSF, linestyle=self.style_DSF, label='DSF', lw=self.linewidth)
        self.progress_line = self.ax_graph.axvline(x=self.df['plot_distance'].iloc[0], color='gray', linestyle=':', lw=1.5)
        
        self.auto_range_axes()

    def auto_range_axes(self):
        if self.df is None: return
        interaction_x_idx = self.df['abs_distance'].idxmin() 
        interaction_x = self.df['plot_car_x'].iloc[interaction_x_idx]
        interaction_y = self.df['ped_y'].iloc[interaction_x_idx]
        view_range_x = max(20, (self.df['plot_car_x'].max() - self.df['plot_car_x'].min()) / 2 + 10)
        view_range_y = max(20, (self.df['car_y'].max() - self.df['car_y'].min()) / 2 + 10)
        self.ax_anim.set_xlim(interaction_x - view_range_x, interaction_x + view_range_x)
        self.ax_anim.set_ylim(interaction_y - view_range_y, interaction_y + view_range_y)
        
        self.ax_graph.set_xlim(self.df['plot_distance'].max(), self.df['plot_distance'].min())
        with np.errstate(invalid='ignore'):
            all_finite_ttc_CRV = np.concatenate([self.ttc_smooth, self.CRV_smooth])
            all_finite_ttc_CRV = all_finite_ttc_CRV[~np.isnan(all_finite_ttc_CRV)]
            self.ax_graph.set_ylim(0, np.nanmax(all_finite_ttc_CRV) * 1.1 if len(all_finite_ttc_CRV) > 0 else 15)
            self.ax_DSF.set_ylim(0, 1.1)

    def setup_animation_and_run(self):
        if self.df is None: return
        if hasattr(self, 'ani_playback') and self.ani_playback and self.ani_playback.event_source: self.ani_playback.event_source.stop()
        
        self.setup_figure_for_animation()
        self.animation_paused = False
        if hasattr(self, 'pause_button'):
            self.pause_button.config(text="Pause Animation")
        
        self.fig.tight_layout(pad=3.0)
        
        pos_graph = self.ax_graph.get_position()
        pos_anim = self.ax_anim.get_position()
        self.ax_anim.set_position([pos_anim.x0, pos_graph.y0, pos_anim.width, pos_graph.height])

        self.ani_playback = animation.FuncAnimation(self.fig, self.animate, frames=len(self.df), interval=50, blit=False, repeat=False)
        self.canvas.draw()
    
    def draw_final_plots(self):
        self.progress_line.set_visible(False)
        
        encounter_dist = 0
        arrival_line = self.ax_graph.axvline(x=encounter_dist, color='purple', linestyle='--', lw=2, label=f'Conflict Point ({encounter_dist:.1f}m)')
        
        self.CRV_line.set_data(self.df['plot_distance'], self.CRV_smooth)
        self.ttc_line.set_data(self.df['plot_distance'], self.ttc_smooth)
        self.DSF_line.set_data(self.df['plot_distance'], self.DSF_smooth)
        
        if self.show_legend:
            self.legend1 = self.ax_graph.legend(handles=[self.CRV_line, self.ttc_line, arrival_line], loc='upper left')
            self.ax_graph.add_artist(self.legend1)
            self.legend2 = self.ax_DSF.legend(handles=[self.DSF_line], loc='upper right')
        
        self.canvas.draw()

    def animate(self, i):
        car_x, car_y = self.df['plot_car_x'][i], self.df['car_y'][i]; ped_x, ped_y = self.df['plot_ped_x'][i], self.df['ped_y'][i]
        angle_rad = np.arctan2(self.df['car_vy_smooth'][i], -self.df['car_vx_smooth'][i])
        self.car_rect.set_transform(transforms.Affine2D().rotate(angle_rad) + transforms.Affine2D().translate(car_x, car_y) + self.ax_anim.transData)
        self.ped_point.set_data([ped_x], [ped_y])
        self.car_trail.set_data(self.df['plot_car_x'][:i+1], self.df['car_y'][:i+1]); self.ped_trail.set_data(self.df['plot_ped_x'][:i+1], self.df['ped_y'][:i+1])
        CRV_vec = self.CRV_vectors[i]; CRV_mag = self.CRV_magnitudes[i]
        speed = np.linalg.norm([self.df['car_vx'][i], self.df['car_vy'][i]])
        if not np.isnan(CRV_mag) and CRV_mag > 1e-6 and speed > 0.1:
            self.CRV_arrow.set_offsets((car_x, car_y)); self.CRV_arrow.set_UVC(-CRV_vec[0], CRV_vec[1])
            self.CRV_text.set_position((car_x - CRV_vec[0]*1.2, car_y + CRV_vec[1]*1.2)); self.CRV_text.set_text(f"{CRV_mag:.2f}")
            self.CRV_arrow.set_visible(True); self.CRV_text.set_visible(True)
        else:
            self.CRV_arrow.set_visible(False); self.CRV_text.set_visible(False)
        ttc_display = f"{self.ttc_values[i]:.2f}s" if not np.isnan(self.ttc_values[i]) else "N/A"; CRV_display = f"{self.CRV_magnitudes[i]:.2f}s" if not np.isnan(self.CRV_magnitudes[i]) else "N/A"; DSF_display = f"{self.DSF_values[i]:.3f}" if not np.isnan(self.DSF_values[i]) else "N/A"
        self.info_text.set_text(f'Time: {self.df["t"][i]:.2f}s | TTC: {ttc_display} | CRV: {CRV_display} | DSF: {DSF_display}')
        
        dist_data = self.df['plot_distance'][:i+1]
        self.CRV_line.set_data(dist_data, self.CRV_smooth[:i+1])
        self.ttc_line.set_data(dist_data, self.ttc_smooth[:i+1])
        self.DSF_line.set_data(dist_data, self.DSF_smooth[:i+1])
        
        current_dist = self.df['plot_distance'][i]
        self.progress_line.set_xdata([current_dist])
        
        if i == len(self.df) - 1: 
            self.draw_final_plots()
            self.status_label.config(text="Animation finished.")
            if hasattr(self, 'pause_button'): self.pause_button.config(state=tk.DISABLED)
        
        return (self.car_rect, self.ped_point, self.car_trail, self.ped_trail, self.CRV_arrow, self.info_text, 
                self.CRV_line, self.ttc_line, self.DSF_line, self.progress_line)

if __name__ == "__main__":
    app = RiskModelVisualizer()
    app.mainloop()
