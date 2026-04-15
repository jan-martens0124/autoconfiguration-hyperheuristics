package examples.ccf;

import AbstractClasses.HyperHeuristic;
import AbstractClasses.ProblemDomain;
import travelingSalesmanProblem.TSP;

/**
 * This class shows how to run the simplified hyper-heuristic with arguments on a selected problem domain.
 * When executing this class, please ensure to set the run configurations. 
 * <id.configuration> <id.instance> <seed> <instance> -d <configurations> -i <configurations> -t <time>
 * 
 * Full list of arguments:
 * id.configuration           : an alphanumeric string as an identifier for a configuration in irace (but is not used within HyFlex)
 * id.instance                : an alphanumeric string as an identifier for an instance in irace (but is not used within HyFlex)
 * seed                       : Seed for the instance generator (default: 1234)
 * instance                   : ID of the problem instance to use
 * -d <configurations>        : Depth of Search (DOS) values separated by spaces (default: 0.2 0.2 0.2)
 * -i <configurations>        : Intensity of Mutation (IOM) values separated by spaces (default: 0.2 0.2 0.2)
 * -t <time>                  : Termination time for the hyper-heuristic in milliseconds
 * 
 * Example program arguments as run configuration: 
 * 1 2024 1234 0 -d 0.1 0.2 0.3 -i 0.4 0.5 0.6 -t 10000
 * 
 * @author Weiyao Meng (weiyao.meng2@nottingham.ac.uk)
 * @date 2024.04.04
 */


public class CCFRunnerConfig {

	public static void main(String[] args) {
		
		// Default values for dos and iom arrays
		double[] dos = {0.2, 0.4, 0.6};
		double[] iom = {0.1, 0.3, 0.5};
		// Default seeds
		long insseed=1234, algseed=5678;
		
		// Flag to track whether -p is specified
		boolean pSpecified = false, tSpecified = false; 
		int insid=0; long time=0; 
		
		//-------------------------------Parse command line arguments for parameters--------------------------
		// Parse command line arguments: <id.configuration> <id.instance> <seed> <instance> -d <configuration> -i <configuration> -t <time>
		
		if(args.length < 4) {
			System.err.println("Usage: java -jar <runner.jar> <id.configuration> <id.instance> <seed> <instance> -d <configuration> -i <configuration> -t <time>");
			System.exit(1);
		}
		insseed = Long.parseLong(args[2]);
		algseed = insseed+1;
		insid = Integer.parseInt(args[3]);
		for (int i = 0; i < args.length; i++) {
		    switch (args[i]) {
	             // termination time
                case "-t":
                	time = Long.parseLong(args[++i]);
                	break;
            	// Parse dos / iom values
		        case "-d":
		        	parseValues(args, dos, i++);
		            break;
		        case "-i":
		        	parseValues(args, iom, i++);
		            break;
		        default:
		            break; // Handle unknown arguments if needed
		    }
		}
		
        //-------------------------------Execute the example HH---------------------------------

		//create a ProblemDomain object with a seed for the random number generator
		ProblemDomain problem = new TSP(insseed);

		//creates an ExampleHyperHeuristic object with a seed for the random number generator
		HyperHeuristic hyper_heuristic_object = new CCF(algseed, dos, iom);

		//we must load an instance within the problem domain, in this case we choose instance 2
		problem.loadInstance(insid);

		//we must set the time limit for the hyper-heuristic in milliseconds, in this example we set the time limit to 1 minute
		hyper_heuristic_object.setTimeLimit(time);

		//a key step is to assign the ProblemDomain object to the HyperHeuristic object. 
		//However, this should be done after the instance has been loaded, and after the time limit has been set
		hyper_heuristic_object.loadProblemDomain(problem);
		
		// Print information before running the hyper-heuristic
//		printAlgorithmInfo(problem, hyper_heuristic_object, time, insid, insseed, algseed); // Note: need to comment this out before exporting a jar


		//now that all of the parameters have been loaded, the run method can be called.
		//this method starts the timer, and then calls the solve() method of the hyper_heuristic_object.
		hyper_heuristic_object.run();
		
		// Obtain the best solution found within the time limit
        System.out.println(hyper_heuristic_object.getBestSolutionValue());

	}
	
	// Helper method to parse DOS and IOM values from command line arguments
    private static void parseValues(String[] args, double[] values, int index) {
        for (int j = 0; j < values.length && index + 1 < args.length; j++) {
            values[j] = Double.parseDouble(args[++index]);
        }
    }
    
 // Print algorithm information before running
    private static void printAlgorithmInfo(ProblemDomain problem, HyperHeuristic hyperHeuristic, long time, int insid, long insseed, long algseed) {
        System.out.println("Algorithm: " + hyperHeuristic.toString());
        System.out.println("Problem instance: " + problem.getClass() + " ID: " + insid);
        System.out.println("Time limit set to: " + time / 1000 + " seconds");
        System.out.println("Instance seed: " + insseed + ". Algorithm seed: " + algseed);
        System.out.println("Search ...");
    }
}