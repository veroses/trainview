export default function Sidebar({ params, onChange, onStart }) {
  return (
    <div className="sidebar">
      <h2 className="sidebar-title">TrainView</h2>
      <button className="start-button" onClick={onStart}>START</button>
      <label>
        Learning Rate:
        <input type="number" name="learning_rate" value={params.learning_rate} onChange={onChange} />
      </label>
      <label>
        Batch Size:
        <input type="number" name="batch_size" value={params.batch_size} onChange={onChange} />
      </label>
      <label>
        Epochs:
        <input type="number" name="epochs" value={params.epochs} onChange={onChange} />
      </label>
      <label>
        Optimizer:
        <select name="optimizer" value={params.optimizer} onChange={onChange}>
          <option value="adam">adam</option>
          <option value="sgd">sgd</option>
        </select>
      </label>
      <div className="sidebar-footer">
        <p>Made by veroses</p>
        <a href="https://github.com" target="_blank" rel="noopener noreferrer">github</a>
      </div>
    </div>
  );
}