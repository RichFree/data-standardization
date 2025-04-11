import { useState } from 'react';

function StringMapper() {
  const [input, setInput] = useState("");
  const [results, setResults] = useState<{ from: string; to: string }[]>([]);

  const handleSubmit = async () => {
    const texts = input
      .split("\n")
      .map(line => line.trim())
      .filter(Boolean); // filters out blank lines

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Accept": "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ texts }),
      });

      // handle bad response - possible 404
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error: ${response.status} — ${errorText}`);
      }

      const data = await response.json();

      // Assuming the response is also a list of strings matching the input
      // ?? is nullish coalescing - if it doesn't work, then move to the next option
      // ? is optional chaining - try to obtain the member
      // explained: if data has no results, return empty list
      const toList = data?.results ?? []; // adjust depending on API response key

      setResults(texts.map((from, idx) => ({
        from,
        to: toList[idx] || "(no match)"
      })));
    } catch (err) {
      console.error("Error during prediction:", err);
      alert("Failed to fetch predictions. See console for details.");
    }
  };

  return (
    <div className="p-4">
      <textarea
        value={input}
        onChange={e => setInput(e.target.value)}
        rows={10}
        className="w-full border p-2"
        placeholder="Enter one string per line"
      />
      <button onClick={handleSubmit} className="mt-2 px-4 py-2 bg-blue-500 text-white rounded">
        Submit
      </button>

      {results.length > 0 && (
        <table className="mt-4 w-full border">
          <thead>
            <tr>
              <th className="border p-2">From</th>
              <th className="border p-2">To</th>
            </tr>
          </thead>
          <tbody>
            {results.map(({ from, to }, idx) => (
              <tr key={idx}>
                <td className="border p-2">{from}</td>
                <td className="border p-2">{to}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}


function App() {
  return (
  <div className="container">
    <StringMapper />
  </div>
  );
}

export default App;
