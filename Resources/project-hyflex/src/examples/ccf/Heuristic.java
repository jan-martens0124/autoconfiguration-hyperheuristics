package examples.ccf;

/** 
 * This class represents a heuristic that can be applied to solve a problem.
 * 
 * @author Warren G Jackson (warren.jackson1@nottingham.ac.uk)
 */

public class Heuristic {

	// Configuration settings for the heuristic
	private final HeuristicConfiguration configuration;
	
	// Data associated with the heuristic
	private final HeuristicData data;
	
	// Unique id for the heuristic
	private final int heuristicId;
	
	/**
     * Constructs a new Heuristic object with the given configuration, ID, and start time.
     * 
     * @param configuration the configuration settings for the heuristic
     * @param heuristicId the unique id for the heuristic
     * @param startTimeNano the start time of the heuristic in nanoseconds
     */
	public Heuristic(HeuristicConfiguration configuration, int heuristicId, long startTimeNano) {
		
		this.configuration = configuration;
		this.data = new HeuristicData(startTimeNano);
		this.heuristicId = heuristicId;
	}

	public HeuristicConfiguration getConfiguration() {
		return configuration;
	}

	public HeuristicData getData() {
		return data;
	}

	public int getHeuristicId() {
		return heuristicId;
	}
}
