package examples.ccf;

import AbstractClasses.HyperHeuristic;
import AbstractClasses.ProblemDomain;
import travelingSalesmanProblem.TSP;

/**
 * Configurable runner for CCF.
 * Supports command-line parameters for irace tuning:
 * <id.configuration> <id.instance> <seed> <instance> -d <dos...> -i <iom...> -p <phi> -t <time>
 */
public class CCFRunnerConfig {

	public static void main(String[] args) {

		// Default seeds and execution settings
		long insseed = 1234, algseed = 5678;
		int insid = 0;
		long time = 60000;
		double phi = 0.50;

		if (args.length < 4) {
			System.err.println("Usage: java -jar <runner.jar> <id.configuration> <id.instance> <seed> <instance> -d <configuration> -i <configuration> -p <phi> -t <time>");
			System.exit(1);
		}

		insseed = Long.parseLong(args[2]);
		algseed = insseed + 1;
		insid = Integer.parseInt(args[3]);

		ProblemDomain problem = new TSP(insseed);
		problem.loadInstance(insid);

		int dosCount = problem.getHeuristicsThatUseDepthOfSearch().length;
		int iomCount = problem.getHeuristicsThatUseIntensityOfMutation().length;
		double[] dos = createDefaultValues(dosCount, 0.2);
		double[] iom = createDefaultValues(iomCount, 0.2);

		for (int i = 4; i < args.length; i++) {
			switch (args[i]) {
				case "-t":
					time = Long.parseLong(args[++i]);
					break;
				case "-p":
					phi = Double.parseDouble(args[++i]);
					break;
				case "-d":
					i = parseValues(args, dos, i + 1);
					break;
				case "-i":
					i = parseValues(args, iom, i + 1);
					break;
				default:
					break;
			}
		}

		HyperHeuristic hyper_heuristic_object = new CCF(algseed, dos, iom, phi);
		hyper_heuristic_object.setTimeLimit(time);
		hyper_heuristic_object.loadProblemDomain(problem);
		hyper_heuristic_object.run();

		// Print only the objective value for irace compatibility
		System.out.println(hyper_heuristic_object.getBestSolutionValue());
	}

	private static int parseValues(String[] args, double[] values, int startIndex) {
		int parsed = 0;
		int i = startIndex;
		while (i < args.length && parsed < values.length && !args[i].startsWith("-")) {
			values[parsed++] = Double.parseDouble(args[i]);
			i++;
		}
		return i - 1;
	}

	private static double[] createDefaultValues(int size, double defaultValue) {
		double[] values = new double[size];
		for (int i = 0; i < size; i++) {
			values[i] = defaultValue;
		}
		return values;
	}
}
