export default function Hyperparams(params, onChange) {
  return (
    <div className="param-row">
      <label>
        epochs:
        <input
          type="number"
          name="epochs"
          value={params.epochs}
          onChange={onChange}
        />
      </label>
      <label>
        learning rate:
        <input
          type="number"
          step="any"
          name="learning_rate"
          value={params.learning_rate}
          onChange={onChange}
        />
      </label>
      <label>
        batch size:
        <input
          type="number"
          name="batch_size"
          value={params.batch_size}
          onChange={onChange}
        />
      </label>
      <label>
        optimizer:
        <select 
        name="optimizer"
        value={params.optimizer}
        onChange={onChange}
        >
        <option value="adam">Adam</option>
        <option value="sgd">SGD</option>
      </select>
      </label>
    </div>
  );
}