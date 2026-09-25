const $ = (selector) => document.querySelector(selector);
const names = ['Compass', 'Field notes', 'Camera', 'Lantern', 'First aid kit', 'Water flask', 'Trail boots', 'Tent', 'Climbing rope', 'Solar charger', 'Supply kit', 'Map case', 'Tool roll', 'Radio', 'Jacket', 'Food pack'];
const glyphs = ['✦', '▤', '◉', '✳', '✚', '◒', '⌁', '△', '◎', '☀', '◫', '◇', '⚒', '◌', '◈', '▥'];
let sourceItems = [];

function fmt(number) { return Number.isInteger(number) ? String(number) : Number(number).toFixed(1); }
function status(message, error = false) { $('#formStatus').textContent = message; $('#formStatus').classList.toggle('error', error); }

function renderInventory() {
  $('#inventoryList').innerHTML = sourceItems.map(([weight, value], i) => `
    <div class="inventory-row">
      <span class="item-name"><span class="item-glyph">${glyphs[i] || '✦'}</span>${names[i] || `Item ${i + 1}`}</span>
      <span>${fmt(weight)} kg</span><span>${value} pts</span>
      <input type="checkbox" data-index="${i}" aria-label="Include ${names[i] || `Item ${i + 1}`}" checked>
    </div>`).join('');
  $('#itemCount').textContent = `${sourceItems.length} ITEMS`;
  $('#inventoryList').addEventListener('change', updatePool);
  updatePool();
}

function updatePool() {
  const selected = document.querySelectorAll('#inventoryList input:checked').length;
  $('#poolCount').textContent = `${selected} / ${sourceItems.length} items`;
  if (sourceItems.length) status('Item pool updated. Train the agent to see the new result.');
}

function drawChart(history, optimum) {
  const max = Math.max(optimum, ...history.map(point => point.best || 0), 1);
  const ceiling = Math.ceil(max / 10) * 10;
  $('#chartMax').textContent = ceiling;
  $('#chartMid').textContent = ceiling / 2;
  $('#chartEnd').textContent = `EPISODE ${history.length}`;
  const points = history.filter((_, i) => i % Math.max(1, Math.ceil(history.length / 140)) === 0 || i === history.length - 1);
  const coordinates = points.map(point => [((point.episode - 1) / Math.max(1, history.length - 1)) * 700, 240 - ((point.best || 0) / ceiling) * 240]);
  const line = coordinates.map(([x, y], i) => `${i ? 'L' : 'M'} ${x.toFixed(1)} ${y.toFixed(1)}`).join(' ');
  $('#linePath').setAttribute('d', line);
  $('#areaPath').setAttribute('d', `${line} L 700 240 L 0 240 Z`);
  $('#chartEmpty').hidden = true;
  $('#chartSummary').textContent = `${history.length} episodes · ${history.filter((point, i) => i && point.best > history[i - 1].best).length} improvements`;
}

function renderResult(result, pool) {
  const { learned, best, optimum, history } = result;
  $('#learnedValue').textContent = learned.value;
  $('#bestValue').textContent = best.value;
  $('#capacityUsed').textContent = fmt(learned.weight);
  $('#capacityCaption').textContent = `of ${fmt(Number($('#capacity').value))} kg available`;
  $('#gapValue').textContent = `${Math.round((optimum.value - learned.value) / Math.max(1, optimum.value) * 100)}%`;
  const selected = learned.indices.map(i => pool[i]);
  $('#loadoutBadge').textContent = `${selected.length} ITEMS PACKED`;
  $('#loadoutList').innerHTML = selected.length ? selected.map(({ index, weight, value }) => `
    <div class="loadout-row"><span class="loadout-glyph">${glyphs[index] || '✦'}</span><div><div class="loadout-name">${names[index] || `Item ${index + 1}`}</div><div class="loadout-details">${fmt(weight)} kg · Item ${String(index + 1).padStart(2, '0')}</div></div><span class="loadout-value">+${value}</span></div>`).join('') : '<div class="empty-loadout">The agent chose to leave the bag empty.</div>';
  const percent = Math.round(learned.weight / Number($('#capacity').value) * 100);
  $('#capacityBar').style.width = `${percent}%`;
  $('#capacityPercent').textContent = `${percent}%`;
  drawChart(history, optimum.value);
  status(`Run complete · ${result.qStates} learned state-action values · exact benchmark ${optimum.value} pts`);
}

async function train(event) {
  if (event) event.preventDefault();
  const checked = [...document.querySelectorAll('#inventoryList input:checked')];
  if (!checked.length) return status('Choose at least one item before training.', true);
  const pool = checked.map(input => {
    const index = Number(input.dataset.index);
    const [weight, value] = sourceItems[index];
    return { index, weight, value };
  });
  const payload = { capacity: Number($('#capacity').value), episodes: Number($('#episodes').value), epsilon: Number($('#epsilon').value) / 100, items: pool.map(({ weight, value }) => [weight, value]) };
  const button = $('#runButton');
  button.disabled = true;
  status('Training in progress…');
  try {
    const response = await fetch('/api/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Training failed.');
    renderResult(result, pool);
  } catch (error) { status(error.message, true); }
  finally { button.disabled = false; }
}

async function init() {
  $('#epsilon').addEventListener('input', event => {
    $('#epsilonValue').textContent = `${event.target.value}%`;
    event.target.style.setProperty('--range', `${event.target.value}%`);
  });
  $('#trainForm').addEventListener('submit', train);
  try {
    const response = await fetch('/api/problem');
    if (!response.ok) throw new Error('Could not load the default problem.');
    const problem = await response.json();
    sourceItems = problem.items;
    $('#capacity').value = problem.capacity;
    renderInventory();
    await train();
  } catch (error) { status(error.message, true); }
}

init();
