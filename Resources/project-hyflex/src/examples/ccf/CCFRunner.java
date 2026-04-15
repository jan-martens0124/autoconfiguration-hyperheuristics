package examples.ccf;

import AbstractClasses.HyperHeuristic;
import AbstractClasses.ProblemDomain;
import travelingSalesmanProblem.TSP;

/**
 * This class shows how to run the simplified hyper-heuristic.

 * @author Weiyao Meng (weiyao.meng2@nottingham.ac.uk)
 * @date 2024.04.04
 */

public class CCFRunner {

	public static void main(String[] args) {
		
		double[] dos = {0.2, 0.4, 0.6};
		double[] iom = {0.1, 0.3, 0.5};

		//create a ProblemDomain object with a seed for the random number generator
		ProblemDomain problem = new TSP(1234);

		//creates an ExampleHyperHeuristic object with a seed for the random number generator
//		HyperHeuristic hyper_heuristic_object = new SCF(5678);
		HyperHeuristic hyper_heuristic_object = new CCF(5678, dos, iom);

		//we must load an instance within the problem domain, in this case we choose instance 2
		problem.loadInstance(0);

		//we must set the time limit for the hyper-heuristic in milliseconds, in this example we set the time limit to 1 minute
		hyper_heuristic_object.setTimeLimit(60000);

		//a key step is to assign the ProblemDomain object to the HyperHeuristic object. 
		//However, this should be done after the instance has been loaded, and after the time limit has been set
		hyper_heuristic_object.loadProblemDomain(problem);
		
		// Print information before running the hyper-heuristic
        System.out.println("Algorithm: "+hyper_heuristic_object.toString());
        System.out.println("Problem instance: " + problem.getClass());
        System.out.println("Time limit set to: " + hyper_heuristic_object.getTimeLimit()/1000 + " seconds");
        System.out.println("Search ...");


		//now that all of the parameters have been loaded, the run method can be called.
		//this method starts the timer, and then calls the solve() method of the hyper_heuristic_object.
		hyper_heuristic_object.run();
		
		// Obtain the best solution found within the time limit
        System.out.println("Best solution value found: " + hyper_heuristic_object.getBestSolutionValue());

	}
}