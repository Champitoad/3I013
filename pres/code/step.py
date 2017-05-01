reward = 0
done = False
info = {'out_of_bounds': False, 'digit': self.current_digit}
# (1)
if not self.position_space.contains(pos):
    info['out_of_bounds'] = True
# (2)
else:
    self.cursor_pos = np.array(pos)
# (3)
if digit < 10:
    if digit != info['digit']:
        reward -= 3
    else:
        reward += 3
        # (4)
        self.cursor_pos = np.array(self.position_space.sample())
# (5)
self.steps += 1
if self.steps >= self.num_steps:
    done = True
# (6)
return self.cursor, reward, done, info
