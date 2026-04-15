package examples.ccf;

public class Heuristic {

	private final HeuristicConfiguration configuration;
	private final int heuristicId;

	public Heuristic(HeuristicConfiguration configuration, int heuristicId) {
		this.configuration = configuration;
		this.heuristicId = heuristicId;
	}

	public HeuristicConfiguration getConfiguration() {
		return configuration;
	}

	public int getHeuristicId() {
		return heuristicId;
	}
}
