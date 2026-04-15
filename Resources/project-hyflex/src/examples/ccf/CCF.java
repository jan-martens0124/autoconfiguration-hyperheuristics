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
	 * creates a new ModifiedChoiceFunctionAllMoves object with a random seed
	 */
	public CCF(long seed){
		super(seed);
	}
	
	/**
	 * This method defines the strategy of the hyper-heuristic
	 * @param problem the problem domain to be solved
	 */
	public void solve(ProblemDomain problem) {  
		
		//record the number of low level heuristics
		int number_of_heuristics = problem.getNumberOfHeuristics();
		
		//initialise phi and delta
		double phi = 0.50, delta = 0.50; 
		//initialise action id, solution quality value etc.
		int action_to_apply = 0, init_flag = 0;
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

		/* 
		 * 'F':  store the calculated scores for each action based on the modified choice function
		 * 'f1': store values related to the performance of actions over time
		 * 'f2': store values related to the relationship between pairs of actions
		 * 'f3': store values related to the time taken to apply each action
		 */
		double[] F = new double[number_of_actions], f1 = new double[number_of_actions], f3 = new double[number_of_actions];
		double[][] f2 = new double[number_of_actions][number_of_actions];
		
		while (!hasTimeExpired()) { //main loop which runs until time has expired
			if (init_flag > 1) { //flag used to select actions randomly for the first two iterations
				// for iterations after the first two
				best_action_score = 0.0;
				
				for (int i = 0; i < number_of_actions; i++) {
					// Update the score for each action using the modified choice function
					F[i] = phi * f1[i] + phi * f2[i][last_action_called] + delta * f3[i];
					// Check if the current action has a better score than the best action so far
					if (F[i] > best_action_score) {
						// If yes, update the best action and its score
						action_to_apply = i; 
						best_action_score = F[i];
					}
				}
			}
			else {
				action_to_apply = rng.nextInt(number_of_actions);
			}
			
			//apply the chosen action to the solution at index 0 in the memory and replace it immediately with the new solution
			Action action = actions.get(action_to_apply);
			time_exp_before = getElapsedTime();
			new_obj_function_value = problem.applyHeuristic(action.first, 0, 0);
			if (action.isChain()) {
				new_obj_function_value = problem.applyHeuristic(action.second, 0, 0);
			}
			time_exp_after = getElapsedTime();
			time_to_apply = time_exp_after - time_exp_before + 1; //+1 prevents / by 0

			//calculate the change in fitness from the current solution to the new solution
			fitness_change = current_obj_function_value - new_obj_function_value;

			//set the current objective function value to the new function value as the new solution is now the current solution
			current_obj_function_value = new_obj_function_value;

			//update f1, f2 and f3 values for appropriate actions 
			//first two iterations dealt with separately to set-up variables
			if (init_flag > 1) {
				f1[action_to_apply] = fitness_change / time_to_apply + phi * f1[action_to_apply];
				f2[action_to_apply][last_action_called] = prev_fitness_change + fitness_change / time_to_apply + phi * f2[action_to_apply][last_action_called];
			} else if (init_flag == 1) {
				f1[action_to_apply] = fitness_change / time_to_apply;
				f2[action_to_apply][last_action_called] = prev_fitness_change + fitness_change / time_to_apply + prev_fitness_change;
				init_flag++;
			} else { //i.e. init_flag = 0
				f1[action_to_apply] = fitness_change / time_to_apply;
				init_flag++;
			} 
			for (int i = 0; i < number_of_actions; i++) {
				f3[i] += time_to_apply;
			}
			f3[action_to_apply] = 0.00;

			if (fitness_change > 0.00) {//in case of improvement
				phi = 0.99;
				delta = 0.01;
				prev_fitness_change = fitness_change / time_to_apply;
			} else {//non-improvement
				if (phi > 0.01) {
					phi -= 0.01;                                                                          
				}
				phi = roundTwoDecimals(phi);
				delta = 1.00 - phi;
				delta = roundTwoDecimals(delta);
				prev_fitness_change = 0.00;
			}
			last_action_called = action_to_apply;
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
	

}
