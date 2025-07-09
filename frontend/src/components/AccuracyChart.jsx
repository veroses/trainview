import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const AccuracyChart = ({ data }) => (
  <LineChart width={400} height={250} data={data}>
    <CartesianGrid strokeDasharray="3 3" />
    <XAxis dataKey="epoch" />
    <YAxis />
    <Tooltip />
    <Line type="monotone" dataKey="accuracy" stroke="#82ca9d" dot={false} />
  </LineChart>
);

export default AccuracyChart;