import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

class RootFileVisualizer:
    def __init__(self, root_file_path):
        self.root_file_path = root_file_path
        self.raw_keys = None
        self.keys = None
        self.event_tree = None
        self.telconfig_tree = None
        self.shower_tree = None
        self.arrayinfo_tree = None
        self.pe_data = None
        self.ntel_data = None
        self.event_number_data = None
        self.tel_az_data = None
        self.tel_al_data = None
        self.sum_pe = None
        self.array_energy = None
        self.array_eventid = None
        self.pix_x_data = None
        self.pix_y_data = None
        self.pix_size_data = None
        self.pix_shape_data = None
        self.focal_length_data = None
        self.tel_id_data = None
        self.num_pixels_data = None
        self.mceventid_data = None
        self.mce0_data = None
        self.mcxcore_data = None
        self.mcycore_data = None
        self.mcze_data = None
        self.mcaz_data = None
        self.xmax_data = None
        self.hini_data = None
        self.paz_data = None
        self.pal_data = None
        self.shower_info_signal = False
        self._load_data()

    def _load_data(self):
        try:
            with uproot.open(self.root_file_path) as root_file:
                self.raw_keys = root_file.keys()
                self.keys = list(set(key.split(";")[0] for key in self.raw_keys))

                event_keys = [key for key in self.keys if key == "true_image" or key.endswith("/true_image")]
                if event_keys:
                    self.event_tree = root_file[event_keys[0]]

                if self.event_tree:
                    npe_keys = [key for key in self.event_tree.keys() if key == "true_pe" or key.endswith("/true_pe")]
                    if npe_keys:
                        self.pe_data = self.event_tree[npe_keys[0]].array(library="np")

                    ntel_keys = [key for key in self.event_tree.keys() if key == "tel_id" or key.endswith("/tel_id")]
                    if ntel_keys:
                        self.ntel_data = self.event_tree[ntel_keys[0]].array(library="np")

                    event_number_keys = [key for key in self.event_tree.keys() if key == "event_id" or key.endswith("/event_id")]
                    if event_number_keys:
                        self.event_number_data = self.event_tree[event_number_keys[0]].array(library="np")

                    tel_az_keys = [key for key in self.event_tree.keys() if key == "tel_az" or key.endswith("/tel_az")]
                    if tel_az_keys:
                        self.tel_az_data = self.event_tree[tel_az_keys[0]].array(library="np")

                    tel_alt_keys = [key for key in self.event_tree.keys() if key == "tel_alt" or key.endswith("/tel_alt")]
                    if tel_alt_keys:
                        self.tel_al_data = self.event_tree[tel_alt_keys[0]].array(library="np")

                    self.sum_pe = np.zeros(len(self.pe_data))
                    for i, event in enumerate(self.pe_data):
                        self.sum_pe[i] = np.sum(event)
                    
                arrayinfo_keys = [key for key in self.keys if key == "arrayevent" or key.endswith("/arrayevent")]
                if arrayinfo_keys:
                    self.arrayinfo_tree = root_file[arrayinfo_keys[0]]

                if self.arrayinfo_tree:
                    array_energy_keys = [key for key in self.arrayinfo_tree.keys() if key == "energy" or key.endswith("/energy")]
                    if array_energy_keys:
                        self.array_energy_data = self.arrayinfo_tree[array_energy_keys[0]].array(library="np")
                        
                    array_eventid_keys = [key for key in self.arrayinfo_tree.keys() if key == "event_id" or key.endswith("/event_id")]
                    if array_eventid_keys:
                        self.array_eventid_data = self.arrayinfo_tree[array_eventid_keys[0]].array(library="np")

                telconfig_keys = [key for key in self.keys if key == "telconfig" or key.endswith("/telconfig")]
                if telconfig_keys:
                    self.telconfig_tree = root_file[telconfig_keys[0]]

                if self.telconfig_tree:
                    tel_id_keys = [key for key in self.telconfig_tree.keys() if key == "tel_id" or key.endswith("/tel_id")]
                    if tel_id_keys:
                        self.tel_id_data = self.telconfig_tree[tel_id_keys[0]].array(library="np")

                    num_pixels_keys = [key for key in self.telconfig_tree.keys() if key == "num_pixels" or key.endswith("/num_pixels")]
                    if num_pixels_keys:
                        self.num_pixels_data = self.telconfig_tree[num_pixels_keys[0]].array(library="np")

                    pix_x_keys = [key for key in self.telconfig_tree.keys() if key == "pix_x" or key.endswith("/pix_x")]
                    if pix_x_keys:
                        self.pix_x_data = self.telconfig_tree[pix_x_keys[0]].array(library="np")

                    pix_y_keys = [key for key in self.telconfig_tree.keys() if key == "pix_y" or key.endswith("/pix_y")]
                    if pix_y_keys:
                        self.pix_y_data = self.telconfig_tree[pix_y_keys[0]].array(library="np")

                    pix_size_keys = [key for key in self.telconfig_tree.keys() if key == "pix_size" or key.endswith("/pix_size")]
                    if pix_size_keys:
                        self.pix_size_data = self.telconfig_tree[pix_size_keys[0]].array(library="np")
                        
                    pix_shape_keys = [key for key in self.telconfig_tree.keys() if key == "pix_shape" or key.endswith("/pix_shape")]
                    if pix_shape_keys:
                        self.pix_shape_data = self.telconfig_tree[pix_shape_keys[0]].array(library="np")

                    focal_length_keys = [key for key in self.telconfig_tree.keys() if key == "focal_length" or key.endswith("/focal_length")]
                    if focal_length_keys:
                        self.focal_length_data = self.telconfig_tree[focal_length_keys[0]].array(library="np")
                        
                    if self.pix_x_data.dtype == 'object':
                        self.pix_x_data = np.array([np.array(row) for row in self.pix_x_data])

                    if self.pix_y_data.dtype == 'object':
                        self.pix_y_data = np.array([np.array(row) for row in self.pix_y_data])

                shower_key = "simulation/shower/shower_info"
                if shower_key in self.keys:
                    self.shower_tree = root_file[shower_key]
                    self.shower_info_signal = True

                    try:
                        self.mceventid_data = self.shower_tree["shower_info/LShower/event_id"].array(library="np")
                        self.mce0_data = self.shower_tree["shower_info/LShower/energy"].array(library="np")
                        self.mcxcore_data = self.shower_tree["shower_info/LShower/core_x"].array(library="np")
                        self.mcycore_data = self.shower_tree["shower_info/LShower/core_y"].array(library="np")
                        self.mcze_data = self.shower_tree["shower_info/LShower/altitude"].array(library="np")
                        self.mcaz_data = self.shower_tree["shower_info/LShower/azimuth"].array(library="np")
                        self.xmax_data = self.shower_tree["shower_info/LShower/x_max"].array(library="np")
                        self.hini_data = self.shower_tree["shower_info/LShower/h_first_int"].array(library="np")
                        self.paz_data = self.shower_tree["shower_info/LShower/array_point_az"].array(library="np")
                        self.pal_data = self.shower_tree["shower_info/LShower/array_point_alt"].array(library="np")
                    except Exception as e:
                        print(f"Error occurred while extracting shower info: {e}")
                        self.shower_info_signal = False

        except Exception as e:
            print(f"Error occurred while loading data: {e}")

    def visualize_event(self, target_event_number=None, max_npe=False, output_path=None):
        if max_npe and self.sum_pe is not None:
            target_event_number = self.event_number_data[np.argmax(self.sum_pe)]

        if target_event_number is None:
            print("Please set a target event ID.")
            return

        indices_with_same_event_number = np.where(self.event_number_data == target_event_number)[0]
        event_id = indices_with_same_event_number[-1]
        ntel_total = len(indices_with_same_event_number)
        tel_num = self.ntel_data[indices_with_same_event_number]
        telconfig_idx = [np.where(self.tel_id_data == t)[0][0] for t in tel_num]

        num_subplots = ntel_total + 1 if self.shower_info_signal else ntel_total
        num_cols = int(np.sqrt(num_subplots))
        num_rows = (num_subplots + num_cols - 1) // num_cols

        telescope_colors = {
            1: 'blue', 2: 'orange', 3: 'green', 4: 'purple',
            5: 'blue', 6: 'orange', 7: 'green', 8: 'purple',
            9: 'blue', 10: 'orange', 11: 'green', 12: 'purple',
            13: 'blue', 14: 'orange', 15: 'green', 16: 'purple'
        }

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols + 2, 6 * num_rows))
        if num_subplots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        colors = ['white'] + plt.cm.plasma(np.linspace(0, 1, 256)).tolist()
        cmap = mcolors.ListedColormap(colors)

        for idx in range(ntel_total):
            ntel = tel_num[idx]
            tel_focal_length = self.focal_length_data[telconfig_idx[idx]]*100
            pixel_numbers = self.num_pixels_data[telconfig_idx[idx]]
            pixel_values = np.zeros(pixel_numbers)

            if len(self.pe_data[indices_with_same_event_number[idx]]) == pixel_numbers:
                pixel_values = self.pe_data[indices_with_same_event_number[idx]]

            max_pe_value = np.max(pixel_values)
            bounds = [0, 1] + np.linspace(1, max_pe_value, 256).tolist()
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            x_positions_cm = self.pix_x_data[telconfig_idx[idx]] * 100
            y_positions_cm = self.pix_y_data[telconfig_idx[idx]] * 100
            square_size = self.pix_size_data[telconfig_idx[idx]] * 100

            axes_idx = idx + 1 if self.shower_info_signal else idx
            self._add_squares(axes[axes_idx], x_positions_cm, y_positions_cm, pixel_values, square_size, norm, cmap)
            axes[axes_idx].set_xlim(min(x_positions_cm) - square_size, max(x_positions_cm) + square_size)
            axes[axes_idx].set_ylim(min(y_positions_cm) - square_size, max(y_positions_cm) + square_size)
            axes[axes_idx].set_aspect('equal')

            axes[axes_idx].set_xlabel('X Position (cm)', fontsize=12)
            axes[axes_idx].set_ylabel('Y Position (cm)', fontsize=12)
            
            secax_x = axes[axes_idx].secondary_xaxis('top', functions=(lambda x: np.degrees(np.arctan(x / tel_focal_length)), lambda x: np.tan(np.radians(x)) * tel_focal_length))
            secax_x.set_xlabel('X Position (degrees)', fontsize=12)

            secax_y = axes[axes_idx].secondary_yaxis('right', functions=(lambda y: np.degrees(np.arctan(y / tel_focal_length)), lambda y: np.tan(np.radians(y)) * tel_focal_length))
            secax_y.set_ylabel('Y Position (degrees)', fontsize=12)

            event_info = f'Telescope {ntel}\nAz: {self.tel_az_data[event_id]:.2f}, Al: {self.tel_al_data[event_id]:.2f}'
            axes[axes_idx].text(0.05, 0.95, event_info, transform=axes[axes_idx].transAxes, fontsize=11, color='black',
                                verticalalignment='top', horizontalalignment='left',
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            divider = make_axes_locatable(axes[axes_idx])
            cax = divider.append_axes("right", size="5%", pad=0.6)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array(np.array([]))
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label('Pe Value', fontsize=12)
            cbar.ax.tick_params(labelsize=10)

        for ax in axes[num_subplots:]:
            ax.axis('off')

        if self.shower_info_signal:
            mceventid_idx = np.where(self.mceventid_data == target_event_number)[0]
            mce0 = self.mce0_data[mceventid_idx].item()
            mcxcore = self.mcxcore_data[mceventid_idx].item()
            mcycore = self.mcycore_data[mceventid_idx].item()
            mcze = self.mcze_data[mceventid_idx].item()
            mcaz = self.mcaz_data[mceventid_idx].item()
            xmax = self.xmax_data[mceventid_idx].item()
            hini = self.hini_data[mceventid_idx].item()

            main_title = (
                f"Event Information\n"
                f"Event ID: {str(target_event_number)}\n"
                f"MCe0: {mce0:.2f}\n"
                f"MCxcore: {mcxcore:.2f}\n"
                f"MCycore: {mcycore:.2f}\n"
                f"MCze: {mcze:.2f}\n"
                f"MCaz: {mcaz:.2f}\n"
                f"xmax: {xmax:.2f}\n"
                f"hini: {hini:.2f}"
            )
            axes[0].axis('off')
            axes[0].text(0.5, 0.5, main_title, fontsize=25, fontweight='bold', ha='center', va='center', transform=axes[0].transAxes)

        plt.tight_layout(pad=1.0)
        if output_path:
            plt.savefig(output_path, dpi=600)
        plt.show()

    def _add_squares(self, ax, x_positions, y_positions, values, square_size, norm, cmap):
        for x, y, value in zip(x_positions, y_positions, values):
            color = cmap(norm(value))
            square = Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size,
                               facecolor=color, edgecolor='k')
            ax.add_patch(square)
    
    def eventID_info(self):
        min_event_id = np.min(self.mceventid_data)
        max_event_id = np.max(self.mceventid_data)

        if self.array_energy_data is not None:
            min_energy_event = self.array_eventid_data[np.argmin(self.array_energy_data)]
            max_energy_event = self.array_eventid_data[np.argmax(self.array_energy_data)]
        else:
            min_energy_event = max_energy_event = None

        min_npe_event = self.event_number_data[np.argmin(self.sum_pe)]
        max_npe_event = self.event_number_data[np.argmax(self.sum_pe)]

        print(f"Event ID Range: {min_event_id} ~ {max_event_id}")
        print("Note that some of Event ID has no trigger image here.")
        print(f"Event ID with Minimum/Maximum Energy: {min_energy_event} / {max_energy_event}")
        print(f"Event ID with Minimum/Maximum NPE: {min_npe_event} / {max_npe_event}")
