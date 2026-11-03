"""pygame operator console: camera previews, status text, keyboard and gamepad input.

Keyboard (hold keys; Shift = slow):
  W/S  +x/-x     A/D  +y/-y     R/F  up/down      (arm base frame)
  Q/E  yaw       T/G  tilt      Z/X  gripper close/open   Space  gripper fully open/closed
  1..9 / Tab     select arm     H    home all arms          P    pause (re-sync to measured)
  Enter          start recording / save episode            Backspace  discard episode
  F1..F9         select task text                          Esc  quit (dataset finalized)

Gamepad (Xbox layout by default, see ``GamepadMap``): left stick x/y, right stick y = z,
right stick x = yaw, D-pad up/down = tilt, LT/RT = gripper close/open, A = gripper toggle,
LB/RB = previous/next arm, Start = record/save, Back = discard, Y = home.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .teleop import Twist


@dataclass
class Speeds:
    lin: float = 0.08          # m/s
    rot_deg: float = 45.0      # deg/s
    grip: float = 0.06         # m/s
    slow: float = 0.3          # factor while Shift is held


@dataclass
class GamepadMap:
    lx: int = 0
    ly: int = 1
    rx: int = 3
    ry: int = 4
    lt: int = 2
    rt: int = 5
    deadzone: float = 0.15
    btn_a: int = 0
    btn_y: int = 3
    btn_lb: int = 4
    btn_rb: int = 5
    btn_back: int = 6
    btn_start: int = 7


@dataclass
class Commands:
    """Edge-triggered operator commands collected during one frame."""
    select_arm: int | None = None
    next_arm: int = 0
    toggle_record: bool = False
    discard: bool = False
    home: bool = False
    pause: bool = False
    quit: bool = False
    toggle_gripper: bool = False
    task: int | None = None


@dataclass
class Console:
    title: str = "piper-vla collect"
    tile: tuple[int, int] = (320, 240)
    speeds: Speeds = field(default_factory=Speeds)
    pad: GamepadMap = field(default_factory=GamepadMap)

    def __post_init__(self) -> None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        if os.environ.get("VLA_AUTOPILOT") or not os.environ.get("DISPLAY"):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless test / no desktop
        import pygame

        self.pg = pygame
        pygame.init()
        pygame.joystick.init()
        self.joy = pygame.joystick.Joystick(0) if pygame.joystick.get_count() else None
        self.screen = None
        self.font = pygame.font.SysFont("monospace", 16)

    # ------------------------------------------------------------------ input

    def poll(self) -> tuple[Twist, Commands]:
        pg = self.pg
        cmd = Commands()
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                cmd.quit = True
            elif ev.type == pg.KEYDOWN:
                k = ev.key
                if k == pg.K_ESCAPE:
                    cmd.quit = True
                elif k in (pg.K_RETURN, pg.K_KP_ENTER):
                    cmd.toggle_record = True
                elif k == pg.K_BACKSPACE:
                    cmd.discard = True
                elif k == pg.K_h:
                    cmd.home = True
                elif k == pg.K_p:
                    cmd.pause = True
                elif k == pg.K_SPACE:
                    cmd.toggle_gripper = True
                elif k == pg.K_TAB:
                    cmd.next_arm += 1
                elif pg.K_1 <= k <= pg.K_9:
                    cmd.select_arm = k - pg.K_1
                elif pg.K_F1 <= k <= pg.K_F9:
                    cmd.task = k - pg.K_F1
            elif ev.type == pg.JOYBUTTONDOWN:
                b = ev.button
                if b == self.pad.btn_start:
                    cmd.toggle_record = True
                elif b == self.pad.btn_back:
                    cmd.discard = True
                elif b == self.pad.btn_y:
                    cmd.home = True
                elif b == self.pad.btn_a:
                    cmd.toggle_gripper = True
                elif b == self.pad.btn_lb:
                    cmd.next_arm -= 1
                elif b == self.pad.btn_rb:
                    cmd.next_arm += 1
            elif ev.type == pg.JOYDEVICEADDED and self.joy is None:
                self.joy = pg.joystick.Joystick(ev.device_index)
        return self._twist(), cmd

    def _twist(self) -> Twist:
        pg, s = self.pg, self.speeds
        keys = pg.key.get_pressed()
        f = s.slow if (keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]) else 1.0
        lin, rot, grip = s.lin * f, np.radians(s.rot_deg) * f, s.grip * f

        def axis(pos, neg):
            return float(keys[pos]) - float(keys[neg])

        tw = Twist(vx=lin * axis(pg.K_w, pg.K_s), vy=lin * axis(pg.K_a, pg.K_d), vz=lin * axis(pg.K_r, pg.K_f),
                   yaw=rot * axis(pg.K_q, pg.K_e), tilt=rot * axis(pg.K_t, pg.K_g),
                   grip=grip * axis(pg.K_x, pg.K_z))
        if self.joy is not None:
            j, m = self.joy, self.pad

            def ax(i):
                if i >= j.get_numaxes():
                    return 0.0
                v = j.get_axis(i)
                return 0.0 if abs(v) < m.deadzone else v

            def trig(i):  # triggers rest at -1 on most pads
                if i >= j.get_numaxes():
                    return 0.0
                v = (j.get_axis(i) + 1.0) / 2.0
                return 0.0 if v < m.deadzone else v

            tw.vx += -lin * ax(m.ly)
            tw.vy += -lin * ax(m.lx)
            tw.vz += -lin * ax(m.ry)
            tw.yaw += -rot * ax(m.rx)
            if j.get_numhats():
                tw.tilt += rot * j.get_hat(0)[1]
            tw.grip += grip * (trig(m.rt) - trig(m.lt))
        return tw

    # ------------------------------------------------------------------ display

    def draw(self, images: list[tuple[str, np.ndarray | None]], lines: list[str], alert: bool = False) -> None:
        pg = self.pg
        tw, th = self.tile
        n = max(1, len(images))
        text_h = 20 * (len(lines) + 1)
        size = (tw * n, th + text_h)
        if self.screen is None or self.screen.get_size() != size:
            self.screen = pg.display.set_mode(size)
            pg.display.set_caption(self.title)
        self.screen.fill((120, 0, 0) if alert else (20, 20, 20))
        for i, (name, img) in enumerate(images):
            if img is not None:
                surf = pg.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
                if surf.get_size() != (tw, th):
                    surf = pg.transform.smoothscale(surf, (tw, th))
                self.screen.blit(surf, (i * tw, 0))
            else:
                self.screen.blit(self.font.render(f"{name}: no image", True, (255, 80, 80)), (i * tw + 8, th // 2))
            self.screen.blit(self.font.render(name, True, (255, 255, 0)), (i * tw + 4, 4))
        for k, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, (230, 230, 230)), (6, th + 6 + 20 * k))
        pg.display.flip()

    def close(self) -> None:
        self.pg.quit()
