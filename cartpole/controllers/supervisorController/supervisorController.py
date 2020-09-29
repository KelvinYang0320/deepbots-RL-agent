import numpy as np
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from PPOAgent import PPOAgent, Transition
from utilities import normalizeToRange

from DQNAgent import DQNAgent
from DoubleDQNAgent import DoubleDQNAgent

class CartPoleSupervisor(SupervisorCSV):
	def __init__(self):
		super().__init__()
		self.observationSpace = 4  # The agent has 4 inputs
		self.actionSpace = 2  # The agent can perform 2 actions
		
		self.robot = None
		self.respawnRobot()
		self.poleEndpoint = self.supervisor.getFromDef("POLE_ENDPOINT")
		self.messageReceived = None	 # Variable to save the messages received from the robot
		
		self.episodeCount = 0  # Episode counter
		self.episodeLimit = 10000  # Max number of episodes allowed
		self.stepsPerEpisode = 200  # Max number of steps per episode
		self.episodeScore = 0  # Score accumulated during an episode
		self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved
		
	def respawnRobot(self):
		if self.robot is not None:
			# Despawn existing robot
			self.robot.remove()

		# Respawn robot in starting position and state
		rootNode = self.supervisor.getRoot()  # This gets the root of the scene tree
		childrenField = rootNode.getField('children')  # This gets a list of all the children, ie. objects of the scene
		childrenField.importMFNode(-2, "CartPoleRobot.wbo")	 # Load robot from file and add to second-to-last position

		# Get the new robot and pole endpoint references
		self.robot = self.supervisor.getFromDef("ROBOT")
		self.poleEndpoint = self.supervisor.getFromDef("POLE_ENDPOINT")
		
	def get_observations(self):
		# Position on z axis, third (2) element of the getPosition vector
		cartPosition = normalizeToRange(self.robot.getPosition()[2], -0.4, 0.4, -1.0, 1.0)
		# Linear velocity on z axis
		cartVelocity = normalizeToRange(self.robot.getVelocity()[2], -0.2, 0.2, -1.0, 1.0, clip=True)
		# Angular velocity x of endpoint
		endpointVelocity = normalizeToRange(self.poleEndpoint.getVelocity()[3], -1.5, 1.5, -1.0, 1.0, clip=True)
		
		# Update self.messageReceived received from robot, which contains pole angle
		self.messageReceived = self.handle_receiver()
		if self.messageReceived is not None:
			poleAngle = normalizeToRange(float(self.messageReceived[0]), -0.23, 0.23, -1.0, 1.0, clip=True)
		else:
			# Method is called before self.messageReceived is initialized
			poleAngle = 0.0
		
		return [cartPosition, cartVelocity, poleAngle, endpointVelocity]
		
	def get_reward(self, action=None):
		return 1
	
	def is_done(self):
		if self.messageReceived is not None:
			poleAngle = round(float(self.messageReceived[0]), 2)
		else:
			# method is called before self.messageReceived is initialized
			poleAngle = 0.0
		if abs(poleAngle) > 0.261799388:  # more than 15 degrees off vertical
			return True

		if self.episodeScore > 195.0:
			return True

		cartPosition = round(self.robot.getPosition()[2], 2)  # Position on z axis
		if abs(cartPosition) > 0.39:
			return True

		return False
	
	def solved(self):
		if len(self.episodeScoreList) > 100:  # Over 100 trials thus far
			if np.mean(self.episodeScoreList[-100:]) > 195.0:  # Last 100 episodes' scores average value
				return True
		return False
		
	def reset(self):
		self.respawnRobot()
		self.supervisor.simulationResetPhysics()  # Reset the simulation physics to start over
		self.messageReceived = None
		return [0.0 for _ in range(self.observationSpace)]
		
	def get_info(self):
		return None


supervisor = CartPoleSupervisor()
agentPPO = PPOAgent(supervisor.observationSpace, supervisor.actionSpace)
agnetDQN = DQNAgent(supervisor.observationSpace, supervisor.actionSpace)
agnetDoubleDQN = DoubleDQNAgent(supervisor.observationSpace, supervisor.actionSpace)
solved = False
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and supervisor.episodeCount < supervisor.episodeLimit:
	observation = supervisor.reset()  # Reset robot and get starting observation
	supervisor.episodeScore = 0
	
	for step in range(supervisor.stepsPerEpisode):
		# In training mode the agentPPO samples from the probability distribution, naturally implementing exploration
		selectedAction, actionProb = agentPPO.work(observation, type_="selectAction")
		selectedActionDQN = agnetDQN.work(observation)
		selectedActionDoubleDQN = agnetDoubleDQN.work(observation)
		selectedAction = selectedActionDoubleDQN
		print("selectedAction:",selectedAction)
		print("selectedDQN:",selectedActionDQN)
		# Step the supervisor to get the current selectedAction's reward, the new observation and whether we reached 
		# the done condition
		newObservation, reward, done, info = supervisor.step([selectedAction])

		# Save the current state transition in agentPPO's memory
		trans = Transition(observation, selectedAction, actionProb, reward, newObservation)
		print("trans:",trans)
		agentPPO.storeTransition(trans)
		agnetDQN.storeTransition(reward, newObservation, done)
		agnetDoubleDQN.storeTransition(reward, newObservation, done)

		if done:
			# Save the episode's score
			supervisor.episodeScoreList.append(supervisor.episodeScore)
			agentPPO.trainStep(batchSize=step)
			solved = supervisor.solved()  # Check whether the task is solved

			agnetDQN.trainStep()
			agnetDoubleDQN.trainStep()
			break

		

		supervisor.episodeScore += reward  # Accumulate episode reward
		observation = newObservation  # observation for next step is current step's newObservation
		
	print("Episode #", supervisor.episodeCount, "score:", supervisor.episodeScore)
	supervisor.episodeCount += 1  # Increment episode counter

if not solved:
	print("Task is not solved, deploying agentPPO for testing...")
elif solved:
	print("Task is solved, deploying agentPPO for testing...")
	
observation = supervisor.reset()
while True:
	selectedAction, actionProb = agentPPO.work(observation, type_="selectActionMax")
	observation, _, _, _ = supervisor.step([selectedAction])
