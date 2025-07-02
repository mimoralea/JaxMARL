import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional
import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class MPEVisualizer(object):
    def __init__(
        self,
        env,
        state_seq: list,
        reward_seq=None,
    ):
        self.env = env

        self.interval = 100
        self.state_seq = state_seq
        self.reward_seq = reward_seq

        # For SimpleSumoMPE, the ring is at index num_agents
        self.ring_index = self.env.num_agents

        self.comm_active = not np.all(self.env.silent)

        self.init_render()

    def animate(
        self,
        save_fname: Optional[str] = None,
        view: bool = True,
        loop: bool = True,
    ):
        """Create an animation of the stored `state_seq`.

        If `loop` is False we append ~30 extra "dummy" frames so the final
        state is shown for a short period before the animation restarts or
        finishes.  This now applies to BOTH on–screen playback and saving so
        behaviour is consistent.
        """
        # Determine how many frames we want to render
        base_frames = len(self.state_seq)
        extra_frames = 30 if not loop else 0
        total_frames = base_frames + extra_frames

        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=total_frames,
            blit=False,
            interval=self.interval,
        )

        # Save the animation if requested
        if save_fname is not None:
            if not loop:
                # A non-looping GIF benefits from the extra freeze frames we have
                # already appended above.  Simply save using the default writer.
                ani.save(save_fname)
            else:
                ani.save(save_fname, loop=0)

        if view:
            plt.show(block=True)

    def init_render(self):
        from matplotlib.patches import Circle
        state = self.state_seq[0]

        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))

        # Get the ring position (should be at (0,0) for SimpleSumoMPE)
        # But we'll get it from the state to be sure
        self.ring_pos = state.p_pos[self.ring_index]

        # Zoom in more to make the ring appear larger (changed from 2 to 0.75)
        ax_lim = 0.75
        self.ax.set_xlim([self.ring_pos[0] - ax_lim, self.ring_pos[0] + ax_lim])
        self.ax.set_ylim([self.ring_pos[1] - ax_lim, self.ring_pos[1] + ax_lim])

        self.entity_artists = []
        self.entity_indices = []  # map artists to entity indices

        # First identify landmarks and agents
        num_agents = self.env.num_agents
        landmark_indices = list(range(num_agents, self.env.num_entities))
        agent_indices = list(range(num_agents))

        # Render landmarks (including the ring) first
        for i in landmark_indices:
            c = Circle(
                state.p_pos[i], self.env.rad[i], color=np.array(self.env.colour[i]) / 255
            )
            self.ax.add_patch(c)
            self.entity_artists.append(c)
            self.entity_indices.append(i)

        # Then render agents on top with consistent colors (green for 'green'/player_0, red for 'red'/player_1)
        for i in agent_indices:
            # Get the agent name to determine color
            agent_name = self.env.agents[i]

            # Map agent names to colors in a robust way
            if agent_name in ("player_0", "green"):
                agent_color = np.array([38, 166, 38]) / 255  # Green
            else:  # "player_1" or "red"
                agent_color = np.array([166, 38, 38]) / 255  # Red

            c = Circle(
                state.p_pos[i], self.env.rad[i], color=agent_color
            )
            self.ax.add_patch(c)
            self.entity_artists.append(c)
            self.entity_indices.append(i)

        # Position text relative to the ring center - adjusted for zoomed-in view
        self.step_counter = self.ax.text(
            self.ring_pos[0] - 0.73,  # Adjusted from -1.95 to fit in zoomed view
            self.ring_pos[1] + 0.73,  # Adjusted from +1.95 to fit in zoomed view
            f"Step: {state.step}",
            va="top",
            fontsize=9  # Smaller font size to fit in the zoomed view
        )

        # --- cumulative reward text for each agent - adjusted for zoomed view ---
        self.rew_texts = []
        if self.reward_seq is not None:
            for i, agent in enumerate(self.env.agents):
                t = self.ax.text(
                    self.ring_pos[0] - 0.73,  # Adjusted from -1.95 to fit in zoomed view
                    self.ring_pos[1] + 0.63 - i * 0.10,  # Adjusted position and spacing
                    f"{agent}: None",
                    va="top",
                    fontsize=8  # Smaller font size for rewards
                )
                self.rew_texts.append(t)

        if self.comm_active:
            self.comm_idx = np.where(self.env.silent == 0)[0]
            self.comm_artists = []
            i = 0
            for idx in self.comm_idx:

                letter = ALPHABET[np.argmax(state.c[idx])]
                a = self.ax.text(self.ring_pos[0] - 1.95, self.ring_pos[1] - 1.95 + i*0.17, f"{self.env.agents[idx]} sends {letter}")

                self.comm_artists.append(a)
                i += 1

    def update(self, frame_index):
        # Make sure we don't go out of bounds if frame is beyond state_seq length
        # This happens when we extend the animation with duplicate last frames
        safe_index = min(len(self.state_seq)-1, frame_index)
        state = self.state_seq[safe_index]
        # Update the ring position for this frame
        current_ring_pos = state.p_pos[self.ring_index]

        # Update each artist using the correct entity index mapping
        for idx, c in zip(self.entity_indices, self.entity_artists):
            c.center = state.p_pos[idx]

        # Update axis limits to keep centered on the ring
        # Maintain the zoomed-in view (same as init_render)
        ax_lim = 0.75
        self.ax.set_xlim([current_ring_pos[0] - ax_lim, current_ring_pos[0] + ax_lim])
        self.ax.set_ylim([current_ring_pos[1] - ax_lim, current_ring_pos[1] + ax_lim])

        # Update text positions to stay relative to the ring - adjusted for zoomed view
        self.step_counter.set_position((current_ring_pos[0] - 0.73, current_ring_pos[1] + 0.73))
        self.step_counter.set_text(f"Step: {safe_index}")

        # Update cumulative reward texts
        if self.reward_seq is not None:
            # Make sure we don't go out of bounds with reward_seq either
            for i, (player, t) in enumerate(zip(self.reward_seq, self.rew_texts)):
                rew_seq = self.reward_seq[player][:safe_index+1]
                t.set_position((current_ring_pos[0] - 0.73, current_ring_pos[1] + 0.63 - i * 0.10))
                t.set_text(
                    f"{player.replace('player_0', 'Green').replace('green', 'Green').replace('player_1', 'Red').replace('red', 'Red')}: {np.sum(rew_seq):.1f}"
                )

        if self.comm_active:
            for i, a in enumerate(self.comm_artists):
                idx = self.comm_idx[i]
                letter = ALPHABET[np.argmax(state.c[idx])]
                # Update communication text position to stay relative to the ring - adjusted for zoomed view
                a.set_position((current_ring_pos[0] - 0.73, current_ring_pos[1] - 0.73 + i*0.10))
                a.set_text(f"{self.env.agents[idx]} sends {letter}")
