package examples.scf;

import AbstractClasses.HyperHeuristic;
import AbstractClasses.ProblemDomain;
import AbstractClasses.ProblemDomain.HeuristicType;


/**
 * This class implements a simplified choice function hyper-heuristics:
 * - heuristic selection: Simplified Choice Function (SCF)
 * - acceptance criteria: a simple 'All Moves' acceptance
 * - low-level heuristics: heuristics of type MUTATION, LOCAL_SEARCH, RUIN_RECREATE.
 * 
 * For each of these heuristics, parameters IOM and DOS are applied.
 * 
 * @author Weiyao Meng (weiyao.meng2@nottingham.ac.uk)
 * @date 2024.03.26
 */

public class SCF extends HyperHeuristic {
	
	// Default values for DOS and IOM parameters
	double[] dosValues = {0.2, 0.2, 0.2}, iomValues = {0.2, 0.2, 0.2}; 
	double phi = 0.50;
	
	
	public SCF(long seed) {
		super(seed);
	}
	
	/**
     * Constructs a new SCF hyper-heuristic with the given seed and custom DOS/IOM values.
     * 
     * @param seed the seed value for random number generation
     * @param dosValue an array of custom DOS values for heuristics
     * @param iomValue an array of custom IOM values for heuristics
     */
	
	public SCF(long seed, double[] dosValue, double[] iomValue) {

		super(seed);
		this.dosValues = dosValue;
		this.iomValues = iomValue;

	}
	
//	public SCF(long seed, double[] dosValue, double[] iomValue, double paramPhi) {
//
//		super(seed);
//		this.dosValues = dosValue;
//		this.iomValues = iomValue;
//		this.phi = paramPhi;
//
//	}
	
	/**
	 * Solves the given problem using the Simplified Choice Function (SCF) hyper-heuristic.
	 * This function implements a loop that continues until the termination criterion is met.
	 * In each iteration, a heuristic is selected either randomly (during initialisation iterations)
	 * or based on the SCF's heuristic selection mechanism. 
	 * The selected heuristic is then applied to the current solution, and its data is updated accordingly. 
	 * The process continues until the termination criterion is reached.
	 * 
	 * @param problem the problem domain to be solved
	 */
	public void solve(ProblemDomain problem) {
		
		int heuristic_to_apply = 0; // Variable to store the ID of the heuristic to apply
		int init_flag = 0; // Flag to track initialisation iterations
		long time_exp_before, time_exp_after, time_to_apply; // Variables for timing
		double new_obj_function_value = 0.00; // Variable to store the objective value of the new solution
		
		// Record the start time in nanoseconds
		long startTimeNano = System.nanoTime();
			
		// Create heuristics array with customised configurations
		Heuristic[] heuristics = createHeuristics(problem, dosValues, iomValues, startTimeNano);
		
		// Determine the set of heuristics to use based on the problem domain
		int[] heuristics_to_use = get_heuristics_to_use(problem);
		
		// Initialise the current solution and get its objective value
		problem.initialiseSolution(0);
		double current_obj_function_value = problem.getFunctionValue(0);

		// Initialise the Simplified Choice Function (SCF) with the created heuristics and customised parameter phi
		SimpleChoiceFunction scf = new SimpleChoiceFunction(heuristics, phi);
//		SimplifiedChoiceFunction scf = new SimplifiedChoiceFunction(heuristics);
		
		// Set the first heuristic to apply
		Heuristic h = heuristics[0];
		
		// Main loop to continue solving until termination criterion is met
		while(!hasTimeExpired()) {
			
			if(init_flag<heuristics_to_use.length) { // Randomly select a heuristic during initialisation
				int randomIndex = rng.nextInt(heuristics_to_use.length);
				heuristic_to_apply = heuristics_to_use[randomIndex];
				h = heuristics[heuristic_to_apply];
				init_flag++;
			}else { // Otherwise, select heuristic using SCF's selection mechanism
				h = scf.selectHeuristicToApply();
			}
			
//			System.out.println("To use next: "+h.getHeuristicId());
			
			// Set the dos and iom based on the selected LLH
			problem.setDepthOfSearch(h.getConfiguration().getDos());
			problem.setIntensityOfMutation(h.getConfiguration().getIom());
						
			// Apply the selected heuristic to the current solution and record the time taken
			time_exp_before = System.nanoTime();
			new_obj_function_value = problem.applyHeuristic(h.getHeuristicId(), 0, 0);
			time_exp_after = System.nanoTime();
			time_to_apply = time_exp_after - time_exp_before + 1; //+1 prevents / by 0
												
			// Update heuristic data based on the applied heuristic and obtained solution
			scf.updateHeuristicData(h, time_exp_before, time_to_apply, current_obj_function_value, new_obj_function_value);
			
			// Print out the information of heuristics
//			this.printHeuristicInfo(h);
//			this.printHeuristicsInfo(heuristics);
			
			//set the current objective function value to the new function value as the new solution is now the current solution
			current_obj_function_value = new_obj_function_value;
		}
	}
	
	public String toString() {

		return "SCF_AM_HH";
	}
	
	/**
	 * Creates an array of heuristic objects based on the problem domain and custom parameter values.
	 * 
	 * @param problem the problem domain
	 * @param dosValues an array of depth of search (DOS) parameter values
	 * @param iomValues an array of intensity of mutation (IOM) parameter values
	 * @param startTimeNano the start time in nanoseconds
	 * @return an array of configured heuristic objects
	 */
	private Heuristic[] createHeuristics(ProblemDomain problem, double[] dosValues, double[] iomValues, long startTimeNano) {
		
		// Initialise array to store heuristic objects
		int numHeuristics = problem.getNumberOfHeuristics();
		Heuristic[] heuristics = new Heuristic[numHeuristics];
		
		// Create heuristic objects with default configurations
    	for (int i = 0; i < numHeuristics; i++) {
            HeuristicConfiguration configuration = new HeuristicConfiguration(0.2, 0.2);
            heuristics[i] = new Heuristic(configuration, i, startTimeNano); 
        }
        
    	// Retrieve heuristics that use DOS and IOM parameters
        int[] dosHeuristics = problem.getHeuristicsThatUseDepthOfSearch();
        int[] iomHeuristics = problem.getHeuristicsThatUseIntensityOfMutation();
 
        // Apply custom DOS values to corresponding heuristic IDs
        for (int i = 0; i < dosHeuristics.length; i++) {
            int id = dosHeuristics[i];
            double value = dosValues[i];
            heuristics[id].getConfiguration().setDos(value);
        }
        
        // Apply custom IOM values to corresponding heuristic IDs
        for (int i = 0; i < iomHeuristics.length; i++) {
            int id = iomHeuristics[i];
            double value = iomValues[i];
            heuristics[id].getConfiguration().setIom(value); 
        }

        return heuristics; //return the array heuristics containing all the configured heuristic objects.
    }
	
	/**
	 * Retrieves an array of heuristic IDs to use based on the problem domain types.
	 * 
	 * @param problem the problem domain
	 * @return an array of heuristic IDs to use
	 */
	private int[] get_heuristics_to_use(ProblemDomain problem) {
		// Retrieve heuristics of different types from the problem domain
		int[] mutations = problem.getHeuristicsOfType(HeuristicType.MUTATION);
		int[] ruin_recreates = problem.getHeuristicsOfType(HeuristicType.RUIN_RECREATE);
		int[] local_searches = problem.getHeuristicsOfType(HeuristicType.LOCAL_SEARCH);
		
		// Calculate total length of all heuristic types
		int totalLength = mutations.length + ruin_recreates.length + local_searches.length;
		
		// Create array to store heuristic IDs to use
		int[] heuristics_to_use = new int[totalLength];
		
		// Copy heuristic IDs of each type to the combined array
		System.arraycopy(mutations, 0, heuristics_to_use, 0, mutations.length);
		System.arraycopy(ruin_recreates, 0, heuristics_to_use, mutations.length, ruin_recreates.length);
		System.arraycopy(local_searches, 0, heuristics_to_use, mutations.length + ruin_recreates.length, local_searches.length);
		
		return heuristics_to_use; // Return the array of heuristic IDs to use
	}
	
	private void printHeuristicInfo(Heuristic h) {
		System.out.println("ID: "+h.getHeuristicId()+" IOM: "+h.getConfiguration().getIom()+" DOS: "+h.getConfiguration().getDos()+" LastApplied: "+h.getData().getTimeLastApplied());
	}
	
	private void printHeuristicsInfo(Heuristic[] heuristics) {
		for(Heuristic h: heuristics) {
			this.printHeuristicInfo(h);
		}
	}
	
	

}
