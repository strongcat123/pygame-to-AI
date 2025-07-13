import pygame
import sys
import random
import math
from collections import defaultdict, deque
from dataclasses import dataclass

# ---------- 상수 ----------
WIDTH, HEIGHT = 800, 600
PATH_WIDTH = 20
N_WAYPOINTS = 3
NUM_BALLS = 10
BALL_RADIUS = 6
SPEED = 3
TURN_ANGLE = 15
FPS = 60
GAMMA = 0.9
ALPHA = 0.12
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.999
REWARD_WAYPOINT = 25
REWARD_FINISH = 250
REWARD_OUT = -75
STEP_PENALTY = -1

# -------------------------------
def get_korean_font(size: int) -> pygame.font.Font:
    candidates = [
        "malgungothic", "Malgun Gothic",
        "NanumGothic", "Nanum Gothic",
        "AppleGothic",
        "Noto Sans CJK KR", "NotoSansCJKkr",
    ]
    for name in candidates:
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.SysFont(None, size)

# ---------- 유틸 함수 ----------
def angle_diff(a, b):
    diff = (a - b + 180) % 360 - 180
    return diff

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ---------- 경로 환경 ----------
class PathWorld:
    def __init__(self):
        margin = 60
        step = (WIDTH - 2 * margin) / (N_WAYPOINTS - 1)
        self.waypoints = []
        for i in range(N_WAYPOINTS):
            x = margin + i * step
            y = random.randint(margin, HEIGHT - margin)
            self.waypoints.append((x, y))

        self.surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for i in range(len(self.waypoints) - 1):
            pygame.draw.line(
                self.surface,
                (50, 200, 50, 255),
                self.waypoints[i],
                self.waypoints[i + 1],
                PATH_WIDTH,
            )
        self.mask = pygame.mask.from_surface(self.surface)

    def heading_to_next(self, pos, idx):
        if idx >= len(self.waypoints):
            # 마지막 포인트 넘어가면 0도 유지
            return 0
        target = self.waypoints[idx]
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        return (math.degrees(math.atan2(-dy, dx)) + 360) % 360

# ---------- 공 데이터 ----------
@dataclass
class Ball:
    pos: list
    heading: float
    idx: int
    finished: bool = False
    trail: deque = None

    def __post_init__(self):
        self.trail = deque(maxlen=10000)  # 충분한 경로 기록

# ---------- Q-러닝 에이전트 ----------
class QLearner:
    def __init__(self):
        self.q = defaultdict(lambda: [0.0, 0.0, 0.0])
        self.epsilon = EPSILON_START

    def discretize(self, world: PathWorld, ball: Ball):
        target_angle = world.heading_to_next(ball.pos, ball.idx)
        ang = angle_diff(target_angle, ball.heading)
        ang_bin = int((ang + 180) // 22.5)
        dist = distance(ball.pos, world.waypoints[min(ball.idx, len(world.waypoints)-1)])
        dist_bin = min(int(dist // 50), 7)
        return (ang_bin, dist_bin)

    def choose(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        values = self.q[state]
        return max(range(3), key=lambda a: values[a])

    def learn(self, s, a, r, s2, done):
        best_next = 0 if done else max(self.q[s2])
        td_target = r + GAMMA * best_next
        self.q[s][a] += ALPHA * (td_target - self.q[s][a])

    def decay_eps(self):
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

# ---------- 시뮬레이터 ----------
class Simulator:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("다중 공 경로 학습")
        self.clock = pygame.time.Clock()
        self.font = get_korean_font(24)

        self.world = PathWorld()
        self.learner = QLearner()
        self.balls = self._spawn_balls()
        self.total_steps = 0

    def _spawn_balls(self):
        balls = []
        for _ in range(NUM_BALLS):
            pos = list(self.world.waypoints[0])
            heading = self.world.heading_to_next(pos, 1)
            balls.append(Ball(pos, heading, 1))
        return balls

    def step_ball(self, ball: Ball, action):
        ball.trail.append(tuple(ball.pos))

        if action == 0:
            ball.heading = (ball.heading + TURN_ANGLE) % 360
        elif action == 2:
            ball.heading = (ball.heading - TURN_ANGLE) % 360

        rad = math.radians(ball.heading)
        ball.pos[0] += SPEED * math.cos(rad)
        ball.pos[1] -= SPEED * math.sin(rad)

        x_i, y_i = int(round(ball.pos[0])), int(round(ball.pos[1]))
        # 경로 밖이면 초기화
        if (x_i < 0 or x_i >= WIDTH or y_i < 0 or y_i >= HEIGHT or
                self.world.mask.get_at((x_i, y_i)) == 0):
            ball.pos = list(self.world.waypoints[0])
            ball.idx = 1
            ball.heading = self.world.heading_to_next(ball.pos, ball.idx)
            ball.trail.clear()
            return REWARD_OUT, False

        if distance(ball.pos, self.world.waypoints[ball.idx]) < PATH_WIDTH / 2:
            ball.idx += 1
            if ball.idx >= len(self.world.waypoints):
                ball.finished = True
                ball.idx = len(self.world.waypoints) - 1
                return REWARD_FINISH, True
            else:
                return REWARD_WAYPOINT, False

        return STEP_PENALTY, False

    def draw(self, freeze=False, winner: Ball | None = None):
        self.screen.fill((30, 30, 30))
        self.screen.blit(self.world.surface, (0, 0))

        for p in self.world.waypoints:
            pygame.draw.circle(self.screen, (255, 255, 255), p, 4)

        if freeze and winner:
            trail_pts = list(winner.trail)
            if len(trail_pts) > 1:
                pygame.draw.lines(self.screen, (150, 255, 255), False, trail_pts, 2)
            pygame.draw.circle(self.screen, (50, 200, 255), (int(winner.pos[0]), int(winner.pos[1])), BALL_RADIUS)
        else:
            for ball in self.balls:
                if len(ball.trail) > 1:
                    pygame.draw.lines(self.screen, (100, 100, 100), False, list(ball.trail), 1)
                color = (200, 50, 50) if not ball.finished else (50, 200, 255)
                pygame.draw.circle(self.screen, color, (int(ball.pos[0]), int(ball.pos[1])), BALL_RADIUS)

        txt = self.font.render(f"스텝: {self.total_steps}  ε: {self.learner.epsilon:.3f}", True, (240, 240, 240))
        self.screen.blit(txt, (10, 10))
        pygame.display.flip()
        self.clock.tick(FPS)

    def run(self):
        running = True
        winner = None
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if winner:
                self.draw(freeze=True, winner=winner)
                continue  # 움직임 정지, 화면 멈춤

            for ball in self.balls:
                if ball.finished:
                    continue
                state = self.learner.discretize(self.world, ball)
                action = self.learner.choose(state)
                reward, done = self.step_ball(ball, action)
                next_state = self.learner.discretize(self.world, ball)
                self.learner.learn(state, action, reward, next_state, done)
                if done and reward == REWARD_FINISH:
                    winner = ball

            self.learner.decay_eps()
            self.total_steps += 1
            self.draw()

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    Simulator().run()