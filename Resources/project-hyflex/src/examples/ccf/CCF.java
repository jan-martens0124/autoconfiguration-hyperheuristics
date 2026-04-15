package examples.ccf;
import AbstractClasses.HyperHeuristic;
import AbstractClasses.ProblemDomain;
import java.text.DecimalFormat;

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
		//initialise heuristic id, solution quality value etc.
		int heuristic_to_apply = 0, init_flag = 0;
		//initialise the variable that stores the ID of the last heuristic that was applied to the solution
		int last_heuristic_called = 0;
		
		//initialise the solution at index 0 in the solution memory array
		problem.initialiseSolution(0); 
		
		//initialise variables which keep track of the objective function values
		double current_obj_function_value = problem.getFunctionValue(0);
		double new_obj_function_value = 0.00, best_heuristic_score = 0.00, fitness_change = 0.00, prev_fitness_change = 0.00;
		
		//initialise variables which keep track of the time usage
		long time_exp_before, time_exp_after, time_to_apply;
		
		/* 
		 * 'F':  store the calculated scores for each heuristic based on the modified choice function
		 * 'f1': store values related to the performance of heuristics over time
		 * 'f2': store values related to the relationship between pairs of heuristics
		 * 'f3': store values related to the time taken to apply each heuristic
		 */
		double[] F = new double[number_of_heuristics], f1 = new double[number_of_heuristics], f3 = new double[number_of_heuristics];
		double[][] f2 = new double[number_of_heuristics][number_of_heuristics];
		
		/*
		 * Retrieve heuristics of type CROSSOVER from the problem domain and assigns a negative infinite value to the corresponding elements of the f3 array
		 * This essentially ensures that heuristics of type CROSSOVER are never selected during the heuristic selection process.
		 */
		int[] crossover_heuristics = problem.getHeuristicsOfType(ProblemDomain.HeuristicType.CROSSOVER);
		for (int i = 0; i < crossover_heuristics.length;i++) {//Give crossover no chance of being selected
			f3[crossover_heuristics[i]]=Double.NEGATIVE_INFINITY;
		}
		
		while (!hasTimeExpired()) { //main loop which runs until time has expired
			if (init_flag > 1) { //flag used to select heuristics randomly for the first two iterations
				// for iterations after the first two
				best_heuristic_score = 0.0;
				
				for (int i = 0; i < number_of_heuristics; i++) {
					// Update the score for each heuristic using the modified choice function
					F[i] = phi * f1[i] + phi * f2[i][last_heuristic_called] + delta * f3[i];
					// Check if the current heuristic has a better score than the best heuristic so far
					if (F[i] > best_heuristic_score) {
						// If yes, update the best heuristic and its score
						heuristic_to_apply = i; 
						best_heuristic_score = F[i];
					}
				}
			}
			else {
				//unpleasant way to check crossover not initially selected randomly
				boolean crossflag = true;
				while(crossflag){
					heuristic_to_apply = rng.nextInt(number_of_heuristics);
					crossflag = false; //assume not crossover before checking if it is
					for (int i = 0; i < crossover_heuristics.length;i++) {
						if(heuristic_to_apply == crossover_heuristics[i]){
							crossflag = true;
						}
					}
				}
			}
			
			//apply the chosen heuristic to the solution at index 0 in the memory and replace it immediately with the new solution
			time_exp_before = getElapsedTime();
			new_obj_function_value = problem.applyHeuristic(heuristic_to_apply, 0, 0);
			time_exp_after = getElapsedTime();
			time_to_apply = time_exp_after - time_exp_before + 1; //+1 prevents / by 0

			//calculate the change in fitness from the current solution to the new solution
			fitness_change = current_obj_function_value - new_obj_function_value;

			//set the current objective function value to the new function value as the new solution is now the current solution
			current_obj_function_value = new_obj_function_value;

			//update f1, f2 and f3 values for appropriate heuristics 
			//first two iterations dealt with separately to set-up variables
			if (init_flag > 1) {
				f1[heuristic_to_apply] = fitness_change / time_to_apply + phi * f1[heuristic_to_apply];
				f2[heuristic_to_apply][last_heuristic_called] = prev_fitness_change + fitness_change / time_to_apply + phi * f2[heuristic_to_apply][last_heuristic_called];
			} else if (init_flag == 1) {
				f1[heuristic_to_apply] = fitness_change / time_to_apply;
				f2[heuristic_to_apply][last_heuristic_called] = prev_fitness_change + fitness_change / time_to_apply + prev_fitness_change;
				init_flag++;
			} else { //i.e. init_flag = 0
				f1[heuristic_to_apply] = fitness_change / time_to_apply;
				init_flag++;
			} 
			for (int i = 0; i < number_of_heuristics; i++) {
				f3[i] += time_to_apply;
			}
			f3[heuristic_to_apply] = 0.00;

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
			last_heuristic_called = heuristic_to_apply;
		}
		
	}
	
	/**
	 * this method must be implemented, to provide a different name for each hyper-heuristic
	 * @return a string representing the name of the hyper-heuristic
	 */
	public String toString() {
		return "Modified Choice Function - All Moves";
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
