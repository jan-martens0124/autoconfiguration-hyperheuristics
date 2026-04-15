package examples.ccf;

/**
 * The simplified version of the modified choice function: 
 * F_t(h_j) = phi_t * f1(h_j) + (1-phi) * f3(h_j)
 * @param F_t: the schore of the heuristic h_j at iteration t.
 * @param f1: a function to score the h_j based on improvement and time taken from its previous application.
 * @param f3: a function to score the h_j based on the time since h_j was last chosen
 * @param phi: if the candidate solution quality was improved, phi = 0.99; else, phi is updated as phi_t = phi_{t-1} - 0.01.
 * 
 * f1(h_j) = I(h_j) / T(h_j)
 * @param I: the change in solution quality as a result of applying h_j
 * @param T: the time taken in seconds to apply the h_j the last time it was applied
 * 
 * f3(h_j) = the time in seconds since the h_j was last chosen
 * 
 * @author Weiyao Meng (weiyao.meng2@nottingham.ac.uk)
 * @date 2024.03.26
 */

public class CustomChoiceFunction {

	// Array of heuristics available for selection
	private Heuristic[] heuristics;

	// Parameter phi used in calculating heuristic scores
	private double phi;

	public CustomChoiceFunction(Heuristic[] heuristics) {

		this.heuristics = heuristics;
		this.phi = 0.50;
	}
	
	public CustomChoiceFunction(Heuristic[] heuristics, double paramPhi) {

		this.heuristics = heuristics;
		this.phi = paramPhi;
	}

	/**
	 * Updates the data associated with the given heuristic based on the provided parameters and the parameter phi.
	 * @param heuristic: the heuristic to update.
	 * @param timeApplied: the current time in nanoseconds.
	 * @param timeTaken: the time taken to apply the heuristic in nanoseconds.
	 * @param current: the objective value of the current solution.
	 * @param candidate: the objective value of the candidate solution.
	 */
	public void updateHeuristicData(Heuristic heuristic, long timeApplied, long timeTaken, double current, double candidate) {
		
		// timeLastApplied
		heuristic.getData().setTimeLastApplied(timeApplied);
		
		// previousApplicationDuration
		heuristic.getData().setPreviousApplicationDuration(timeTaken);
		
		// F_delta
		heuristic.getData().setF_delta(current-candidate); // if improvement - f_delta is negative
		
		// update phi based on improvement/non-improvement
		this.updatePhi(current, candidate);
		
	}

	/**
     * Selects and returns the heuristic to apply based on calculated scores.
     * 
     * @return the selected heuristic to apply
     */
	public Heuristic selectHeuristicToApply() {
		Heuristic selectedHeuristic = null;
		double bestScore = -Double.MAX_VALUE;
		long currentTime = System.nanoTime();
		for (Heuristic heuristic : heuristics) {
//			System.out.print("Score of "+heuristic.getHeuristicId()+" is:");
            double score = this.calculateScore(heuristic, currentTime);
            if (score > bestScore) {
                bestScore = score;
                selectedHeuristic = heuristic;
            }
//            System.out.println(" Score: "+score);
        }
		return selectedHeuristic;
	}
	
	/**
     * Calculates the score for the given heuristic based on specific criteria.
     * 
     * @param h the heuristic for which the score is calculated
     * @param currentTime the current time in nanoseconds
     * @return the calculated score for the heuristic
     */
	public double calculateScore(Heuristic h, long currentTime) {
		
		// calculate f1
		double I = h.getData().getF_delta();
		double T = (h.getData().getPreviousApplicationDuration()/1_000_000_000)+1;
		double f1 = I/T;
		
		// calculate f3
		double f3 = (currentTime - h.getData().getTimeLastApplied())/ 1_000_000_000;
		
//		System.out.print("f1 score: "+f1+" f3 score: "+f3);
		
		// calculate overall score F
		return this.phi*f1+(1-phi)*f3;
		
	}

	/**
     * Updates the phi parameter based on improvement or non-improvement.
     * 
     * @param current the objective value of the current solution
     * @param candidate the objective value of the candidate solution
     */
	public void updatePhi(double current, double candidate) {
		if(current>candidate) {//in case of improvement
			this.phi = 0.99;
		}else {//non-improvement
			this.phi -= 0.01;

		}
	}
	

}
