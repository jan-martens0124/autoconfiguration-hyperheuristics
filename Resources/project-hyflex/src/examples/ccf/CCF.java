package examples.ccf;
import AbstractClasses.HyperHeuristic;
import AbstractClasses.ProblemDomain;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * This is the source code of modified choice function using a simple 'All Moves' acceptance criteria as described in: 
 * Drake, J. H., Özcan, E., & Burke, E. K. (2012). An improved choice function heuristic selection for cross domain heuristic search. 
 * Original source code available at https://www.researchgate.net/publication/257922567_Java_Source_of_Modified_Choice_Function_-_All_Moves_hyper-heuristic
 * 
 * This version has been adapted for use in the NATCOR2024 @UoN by @author Weiyao Meng (weiyao.meng2@nottingham.ac.uk)
 * Additional comments have been added for clarity.
 * 
 * @date 2024.03.26
 */

public class CCF extends HyperHeuristic {

	// Default values for DOS and IOM parameters
	double[] dosValues = {0.2, 0.2, 0.2}, iomValues = {0.2, 0.2, 0.2}; 

	// Default value for phi parameter
	double phi = 0.50, delta = 0.50;

	private static class Action {
		private final int first;
		private final Integer second;

		Action(int first) {
			this.first = first;
			this.second = null;
		}

		Action(int first, int second) {
			this.first = first;
			this.second = second;
		}

		boolean isChain() {
			return second != null;
		}
	}
	
	/**
	 * creates a new CustomChoiceFunctionAllMoves object with a random seed
	 */
	public CCF(long seed){
		super(seed);
	}

	/**
     * Constructs a new CCF hyper-heuristic with the given seed and custom DOS/IOM values.
     * 
     * @param seed the seed value for random number generation
     * @param dosValue an array of custom DOS values for heuristics
     * @param iomValue an array of custom IOM values for heuristics
     * @param phi the phi parameter for the modified choice function
     */
	public CCF(long seed, double[] dosValues, double[] iomValues, double phi){
		super(seed);
		this.dosValues = dosValues;
		this.iomValues = iomValues;
		this.phi = phi;
		this.delta = 1.00 - phi;
	}

	/**
	 * This method defines the strategy of the hyper-heuristic
	 * @param problem the problem domain to be solved
	 */
	public void solve(ProblemDomain problem) {  
		
		//record the number of low level heuristics
		int number_of_heuristics = problem.getNumberOfHeuristics();
		Heuristic[] heuristics = createHeuristics(problem, dosValues, iomValues);
		final int tournamentSize = 3;
		
		//initialise phi and delta
		double phi = this.phi, delta = this.delta; 
		//initialise action id, solution quality value etc.
		int best_action_to_apply = 0, init_flag = 0;
		//initialise the variable that stores the ID of the last action that was applied to the solution
		int last_action_called = 0;
		
		//initialise the solution at index 0 in the solution memory array
		problem.initialiseSolution(0); 
		
		//initialise variables which keep track of the objective function values
		double current_obj_function_value = problem.getFunctionValue(0);
		double new_obj_function_value = 0.00, best_action_score = 0.00, fitness_change = 0.00, prev_fitness_change = 0.00;
		
		//initialise variables which keep track of the time usage
		long time_exp_before, time_exp_after, time_to_apply;
		
		/*
		 * Retrieve heuristics of type CROSSOVER from the problem domain.
		 * This ensures that heuristics of type CROSSOVER are never selected in either single or chained actions.
		 */
		int[] crossover_heuristics = problem.getHeuristicsOfType(ProblemDomain.HeuristicType.CROSSOVER);
		boolean[] isCrossover = new boolean[number_of_heuristics];
		for (int i = 0; i < crossover_heuristics.length; i++) {
			isCrossover[crossover_heuristics[i]] = true;
		}

		/*
		 * Build action space:
		 * - single LLH actions: (h)
		 * - chained LLH actions: (h1, h2)
		 */
		List<Action> actions = new ArrayList<>();
		for (int i = 0; i < number_of_heuristics; i++) {
			if (!isCrossover[i]) {
				actions.add(new Action(i));
			}
		}
		for (int i = 0; i < number_of_heuristics; i++) {
			if (isCrossover[i]) {
				continue;
			}
			for (int j = 0; j < number_of_heuristics; j++) {
				if (!isCrossover[j]) {
					actions.add(new Action(i, j));
				}
			}
		}
		int number_of_actions = actions.size();
		if (number_of_actions == 0) {
			return;
		}

		// 0 is current solution; 1,2,3 are trial solutions for candidate actions
		problem.setMemorySize(4);

		/* 
		 * 'F':  store the calculated scores for each action based on the modified choice function
		 * 'f1': store values related to the performance of actions over time
		 * 'f2': store values related to the relationship between pairs of actions
		 * 'f3': store values related to the time taken to apply each action
		 */
		double[] F = new double[number_of_actions], f1 = new double[number_of_actions], f3 = new double[number_of_actions];
		double[][] f2 = new double[number_of_actions][number_of_actions];
		
		while (!hasTimeExpired()) { //main loop which runs until time has expired
			best_action_score = 0.0;

			for (int i = 0; i < number_of_actions; i++) {
				// Update the score for each action using the modified choice function
				F[i] = phi * f1[i] + phi * f2[i][last_action_called] + delta * f3[i];
				// Check if the current action has a better score than the best action so far
				if (F[i] > best_action_score) {
					// If yes, update the best action and its score
					best_action_to_apply = i;
					best_action_score = F[i];
				}
			}

			int candidatesToTry = number_of_actions >= 3 ? 2 + rng.nextInt(2) : Math.min(2, number_of_actions);
			int[] candidateActions = new int[candidatesToTry];
			boolean[] chosen = new boolean[number_of_actions];
			int selectedCount = 0;

			candidateActions[selectedCount++] = best_action_to_apply;
			chosen[best_action_to_apply] = true;

			while (selectedCount < candidatesToTry) {
				int selected = selectByTournament(F, chosen, tournamentSize);
				if (selected < 0) {
					break;
				}
				candidateActions[selectedCount++] = selected;
				chosen[selected] = true;
			}

			int winningAction = candidateActions[0];
			double winningObjValue = Double.POSITIVE_INFINITY;
			long winningTimeToApply = 1;
			long totalTrialTime = 0;

			// Evaluate candidates on copied solutions and accept the most effective one
			for (int c = 0; c < selectedCount; c++) {
				int actionIndex = candidateActions[c];
				Action action = actions.get(actionIndex);
				int targetIndex = c + 1; // use memory slots 1..3

				time_exp_before = getElapsedTime();
				new_obj_function_value = executeActionOnMemory(problem, heuristics, action, 0, targetIndex);
				time_exp_after = getElapsedTime();
				time_to_apply = time_exp_after - time_exp_before + 1; //+1 prevents / by 0
				totalTrialTime += time_to_apply;

				if (new_obj_function_value < winningObjValue) {
					winningObjValue = new_obj_function_value;
					winningAction = actionIndex;
					winningTimeToApply = time_to_apply;
				}
			}

			for (int c = 0; c < selectedCount; c++) {
				if (candidateActions[c] == winningAction) {
					problem.copySolution(c + 1, 0);
					break;
				}
			}

			//calculate the change in fitness from the current solution to the new accepted solution
			fitness_change = current_obj_function_value - winningObjValue;
			current_obj_function_value = winningObjValue;

			//update f1, f2 and f3 values for appropriate actions 
			//first two iterations dealt with separately to set-up variables
			if (init_flag > 1) {
				f1[winningAction] = fitness_change / winningTimeToApply + phi * f1[winningAction];
				f2[winningAction][last_action_called] = prev_fitness_change + fitness_change / winningTimeToApply + phi * f2[winningAction][last_action_called];
			} else if (init_flag == 1) {
				f1[winningAction] = fitness_change / winningTimeToApply;
				f2[winningAction][last_action_called] = prev_fitness_change + fitness_change / winningTimeToApply + prev_fitness_change;
				init_flag++;
			} else { //i.e. init_flag = 0
				f1[winningAction] = fitness_change / winningTimeToApply;
				init_flag++;
			} 
			for (int i = 0; i < number_of_actions; i++) {
				f3[i] += totalTrialTime;
			}
			f3[winningAction] = 0.00;

			if (fitness_change > 0.00) {//in case of improvement
				phi = 0.99;
				delta = 0.01;
				prev_fitness_change = fitness_change / winningTimeToApply;
			} else {//non-improvement
				if (phi > 0.01) {
					phi -= 0.01;                                                                          
				}
				phi = roundTwoDecimals(phi);
				delta = 1.00 - phi;
				delta = roundTwoDecimals(delta);
				prev_fitness_change = 0.00;
			}
			last_action_called = winningAction;
		}
		
	}
	
	/**
	 * this method must be implemented, to provide a different name for each hyper-heuristic
	 * @return a string representing the name of the hyper-heuristic
	 */
	public String toString() {
		return "Custom Choice Function - All Moves (with LLH chaining)";
	}
	
	/**
	 * this method is introduced to combat some rounding issues introduced by Java
	 * @return a double of the input d, rounded to two decimal places
	 */
	public double roundTwoDecimals(double d) {
		DecimalFormat two_d_form = new DecimalFormat("#.##");
		return Double.valueOf(two_d_form.format(d));
	}

	private void applyHeuristicWithConfiguration(ProblemDomain problem, Heuristic heuristic) {
		problem.setDepthOfSearch(heuristic.getConfiguration().getDos());
		problem.setIntensityOfMutation(heuristic.getConfiguration().getIom());
	}

	private double executeActionOnMemory(ProblemDomain problem, Heuristic[] heuristics, Action action, int sourceIndex, int targetIndex) {
		applyHeuristicWithConfiguration(problem, heuristics[action.first]);
		double result = problem.applyHeuristic(action.first, sourceIndex, targetIndex);
		if (action.isChain()) {
			applyHeuristicWithConfiguration(problem, heuristics[action.second]);
			result = problem.applyHeuristic(action.second, targetIndex, targetIndex);
		}
		return result;
	}

	private int selectByTournament(double[] scores, boolean[] excluded, int tournamentSize) {
		int n = scores.length;
		int available = 0;
		for (int i = 0; i < n; i++) {
			if (!excluded[i]) {
				available++;
			}
		}
		if (available == 0) {
			return -1;
		}

		int samples = Math.min(tournamentSize, available);
		int best = -1;
		double bestScore = Double.NEGATIVE_INFINITY;
		boolean[] localPicked = new boolean[n];

		for (int s = 0; s < samples; s++) {
			int idx = -1;
			while (idx < 0 || excluded[idx] || localPicked[idx]) {
				idx = rng.nextInt(n);
			}
			localPicked[idx] = true;
			if (scores[idx] > bestScore) {
				bestScore = scores[idx];
				best = idx;
			}
		}
		return best;
	}

	private Heuristic[] createHeuristics(ProblemDomain problem, double[] dosValues, double[] iomValues) {
		int numHeuristics = problem.getNumberOfHeuristics();
		Heuristic[] heuristics = new Heuristic[numHeuristics];

		for (int i = 0; i < numHeuristics; i++) {
			HeuristicConfiguration configuration = new HeuristicConfiguration(0.2, 0.2);
			heuristics[i] = new Heuristic(configuration, i);
		}

		int[] dosHeuristics = problem.getHeuristicsThatUseDepthOfSearch();
		int[] iomHeuristics = problem.getHeuristicsThatUseIntensityOfMutation();

		int dosLimit = Math.min(dosHeuristics.length, dosValues.length);
		for (int i = 0; i < dosLimit; i++) {
			int id = dosHeuristics[i];
			heuristics[id].getConfiguration().setDos(dosValues[i]);
		}

		int iomLimit = Math.min(iomHeuristics.length, iomValues.length);
		for (int i = 0; i < iomLimit; i++) {
			int id = iomHeuristics[i];
			heuristics[id].getConfiguration().setIom(iomValues[i]);
		}

		return heuristics;
	}
	

}
