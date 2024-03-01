from typing import Hashable, Tuple
import tkinter as tk
from tkinter.filedialog import askopenfilename
from domains.simulator import Simulator


# TODO: game configuration ui / logic
class AppInterface():

  def __init__(self) -> None:
    self.main_window = None
    self.game = None  # type: Simulator
    self.canvas_width = 300
    self.canvas_height = 300
    self.__init_window()
    self._init_game()

    self._replay_mode = False
    self._started = False
    self._loaded = False
    self._event_based = True
    self._trajectory = None
    self._current_step = 0

  def run(self):
    self.main_window.mainloop()

  def _init_game(self):
    'define game related variables and objects'
    # self.game = Simulator()
    pass

  def __init_window(self):
    self.main_window = tk.Tk()

    self._init_gui()

  def _init_gui(self):
    self.btn_start = tk.Button(self.main_window,
                               text="Start",
                               command=self._on_start_btn_clicked)
    self.btn_start.grid(row=0, column=1)

    self.btn_reset = tk.Button(self.main_window,
                               text="Reset",
                               command=self._on_reset_btn_clicked)
    self.btn_reset.grid(row=1, column=1)

    self.label_score = tk.Label(self.main_window, text="0")
    self.label_score.grid(row=2, column=1)

    self.canvas = tk.Canvas(self.main_window,
                            width=self.canvas_width,
                            height=self.canvas_height,
                            highlightbackground="black")
    self.canvas.grid(row=1, rowspan=2)
    self.canvas.bind("<Key>", self._on_key_pressed)
    self.canvas.bind("<Button-1>", self._on_mouse_l_btn_clicked)
    self.canvas.bind("<Button-2>", self._on_mouse_r_btn_clicked)
    self.canvas.focus_set()

  # [New game, Replay],  Close
  #   agent1 : [user, ai]
  #   agent2 : [user, ai]
  #   ...
  #   Period-based / Event-based
  #   Start / Pause,  Reset
  #   -----
  #   Load / Clear,  File name
  #   Auto play / Pause, Reset
  #   -----
  #   Canvas / Scene
  #   -----
  #   Parameters
  #   Log

  def _on_mode_selected(self, idx: int):
    if idx == 0:
      self._replay_mode = False
    else:
      self._replay_mode = True

    self.__update_mode_ui()

  def __update_mode_ui(self):
    if self._replay_mode:
      pass
    else:
      pass

  def _on_close_btn_clicked(self):
    pass

  def _on_reset_btn_clicked(self):
    if self._replay_mode:
      self._current_step = 0
      pass
    else:
      self.game.reset_game()
      self._update_ctrl_ui()
      self._update_canvas_scene()
      self._update_canvas_overlay()

    # clean canvas

  def _on_start_btn_clicked(self):
    if self._replay_mode:
      return

    self._started = not self._started
    # change the button text
    if self._started:
      self.btn_start.config(text="Pause")
    else:
      self.btn_start.config(text="Start")

    # run
    if self._event_based:
      if self._started:
        # ready for inputs
        self._update_ctrl_ui()
        self._update_canvas_scene()
        self._update_canvas_overlay()
        pass
      else:
        # not take inputs
        pass

  def _on_load_btn_clicked(self):
    self._loaded = not self._loaded
    # change the button text

    # open file selector
    if self._loaded:
      file_name = askopenfilename()
      if file_name == '':
        return

      self._trajectory = self.game.read_file(file_name)
      self._current_step = 0
    else:
      self._trajectory = None

  def _conv_key_to_agent_event(self,
                               key_sym) -> Tuple[Hashable, Hashable, Hashable]:
    pass

  def _conv_mouse_to_agent_event(
      self, is_left: bool,
      cursor_pos: Tuple[float, float]) -> Tuple[Hashable, Hashable, Hashable]:
    pass

  def _on_key_pressed(self, key_event):
    if not self._started:
      return

    agent, e_type, e_value = self._conv_key_to_agent_event(key_event.keysym)
    self.game.event_input(agent, e_type, e_value)
    if self._event_based:
      action_map = self.game.get_joint_action()
      self.game.take_a_step(action_map)

      if not self.game.is_finished():
        # update canvas
        self._update_ctrl_ui()
        self._update_canvas_scene()
        self._update_canvas_overlay()
        # pop-up for latent?
      else:
        self._on_game_end()
    else:
      pass

  def _on_mouse_l_btn_clicked(self, event):
    if not self._started:
      return

    LEFT_CLICK = True
    agent, e_type, e_value = self._conv_mouse_to_agent_event(
        LEFT_CLICK, (event.x, event.y))
    self.game.event_input(agent, e_type, e_value)

  def _on_mouse_r_btn_clicked(self, event):
    if not self._started:
      return

    RIGHT_CLICK = False
    agent, e_type, e_value = self._conv_mouse_to_agent_event(
        RIGHT_CLICK, (event.x, event.y))
    self.game.event_input(agent, e_type, e_value)

  def _update_ctrl_ui(self):
    self.label_score.config(text=str(self.game.get_score()))

  def _update_canvas_scene(self):
    pass

  def _update_canvas_overlay(self):
    pass

  def _on_game_end(self):
    pass

  def clear_canvas(self):
    self.canvas.delete("all")

  def create_rectangle(self, x_pos_st, y_pos_st, x_pos_ed, y_pos_ed, color):
    return self.canvas.create_rectangle(x_pos_st,
                                        y_pos_st,
                                        x_pos_ed,
                                        y_pos_ed,
                                        fill=color)

  def create_line(self, x_pos_st, y_pos_st, x_pos_ed, y_pos_ed, color, width=5):
    return self.canvas.create_line(x_pos_st,
                                   y_pos_st,
                                   x_pos_ed,
                                   y_pos_ed,
                                   fill=color,
                                   width=width)

  def create_oval(self, x_pos_st, y_pos_st, x_pos_ed, y_pos_ed, color):
    return self.canvas.create_oval(x_pos_st,
                                   y_pos_st,
                                   x_pos_ed,
                                   y_pos_ed,
                                   fill=color)

  def create_circle(self, x_cen, y_cen, radius, color):
    x_pos_st = (x_cen - radius)
    x_pos_ed = (x_cen + radius)
    y_pos_st = (y_cen - radius)
    y_pos_ed = (y_cen + radius)
    return self.canvas.create_oval(x_pos_st,
                                   y_pos_st,
                                   x_pos_ed,
                                   y_pos_ed,
                                   fill=color)

  def create_text(self, x_center, y_center, txt, color='black'):
    return self.canvas.create_text(x_center, y_center, text=txt, fill=color)

  def create_triangle(self, x_st, y_st, width, height, color):
    x_pos_1 = x_st
    y_pos_1 = y_st

    x_pos_2 = x_st + width
    y_pos_2 = y_st

    x_pos_3 = (x_pos_1 + x_pos_2) / 2
    y_pos_3 = y_st + height
    points = [x_pos_1, y_pos_1, x_pos_3, y_pos_3, x_pos_2, y_pos_2]

    return self.canvas.create_polygon(points, fill=color)
