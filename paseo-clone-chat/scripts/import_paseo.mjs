#!/usr/bin/env node
// Reuse the installed Paseo CLI's authenticated connection and workspace API.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const { values: args } = parseArgs({ options: {
  'cli-root': { type: 'string' }, host: { type: 'string' },
  'source-agent': { type: 'string' }, state: { type: 'string' },
  title: { type: 'string' }, 'existing-agent': { type: 'string' },
  check: { type: 'boolean', default: false }, help: { type: 'boolean' },
} });
if (args.help) {
  console.log('import_paseo.mjs --cli-root DIR --host ENDPOINT --source-agent ID --state FILE --title NAME [--existing-agent ID] [--check]');
  process.exit(0);
}
for (const key of ['cli-root', 'host', 'source-agent', 'state', 'title']) {
  assert.ok(args[key]?.trim(), `Missing --${key}`);
}
const root = path.resolve(args['cli-root']);
assert.equal(JSON.parse(fs.readFileSync(path.join(root, 'package.json'))).name, '@getpaseo/cli');
const state = JSON.parse(fs.readFileSync(args.state, 'utf8'));
assert.ok(['history_verified', 'import_requested', 'imported', 'complete'].includes(state.status), 'History has not been verified');
assert.deepEqual(state.source_proof, state.copy_proof, 'History proof differs');
assert.ok(state.source_proof?.turn_count > 0, 'History proof is missing');
assert.notEqual(state.copy_id, state.source_id);
assert.ok(state.copy_id, 'Copy ID is missing');
const existingId = args['existing-agent'] || state.new_agent_id;
if (args.check || state.status === 'import_requested') {
  assert.ok(existingId, 'Reconcile the previous import and provide --existing-agent; do not import again');
}
const save = () => {
  const temporary = `${args.state}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, JSON.stringify(state, null, 2) + '\n', { mode: 0o600, flag: 'wx' });
  fs.renameSync(temporary, args.state);
  assert.deepEqual(JSON.parse(fs.readFileSync(args.state, 'utf8')), state);
};
const { connectToDaemon } = await import(pathToFileURL(path.join(root, 'dist/utils/client.js')));
const client = await connectToDaemon({ host: args.host, timeout: 15000 });
const deadline = setTimeout(() => {
  console.error('Import exceeded 120 seconds; reconcile saved state before retrying.');
  process.exit(1);
}, 120000);
try {
  const source = (await client.fetchAgent({ agentId: args['source-agent'] })).agent;
  assert.equal(source.provider, 'codex');
  assert.equal(source.persistence.sessionId, state.source_id);
  assert.equal(source.cwd, state.cwd);
  assert.ok(source.workspaceId, 'Source workspace is missing');
  const config = {};
  const model = source.runtimeInfo?.model || source.model;
  const thinking = source.effectiveThinkingOptionId ?? source.thinkingOptionId;
  if (model) config.modelId = model;
  if (thinking) config.thinkingOptionId = thinking;
  if (source.currentModeId) config.modeId = source.currentModeId;
  const features = Object.fromEntries((source.features || [])
    .filter(f => f.type === 'toggle' && typeof f.value === 'boolean' && !f.disabled)
    .map(f => [f.id, f.value]));
  if (Object.keys(features).length) config.featureValues = features;
  if (!args.check) {
    state.source_agent_id = source.id;
    state.workspace_id = source.workspaceId;
    if (!existingId) {
      state.status = 'import_requested'; save();
      const imported = await client.importAgent({ provider: 'codex', sessionId: state.copy_id,
        cwd: state.cwd, workspaceId: source.workspaceId });
      state.new_agent_id = imported.id; state.status = 'imported'; save();
    } else {
      state.new_agent_id = existingId;
    }
  }
  const agentId = existingId || state.new_agent_id;
  const inspect = async () => (await client.fetchAgent({ agentId })).agent;
  let copy = await inspect();
  assert.notEqual(copy.id, source.id);
  assert.equal(copy.persistence.sessionId, state.copy_id);
  assert.equal(copy.workspaceId, source.workspaceId);
  assert.equal(copy.status, 'idle', 'Copy is active; do not change its settings');
  if (!args.check) {
    await client.applyAgentConfig(agentId, config);
    await client.updateAgent(agentId, { name: args.title });
    copy = await inspect();
  }
  assert.equal(copy.title, args.title);
  assert.equal(copy.status, 'idle');
  if (model) assert.equal(copy.runtimeInfo?.model || copy.model, model);
  if (thinking) assert.equal(copy.effectiveThinkingOptionId, thinking);
  if (config.modeId) assert.equal(copy.currentModeId, config.modeId);
  for (const [id, value] of Object.entries(features)) {
    assert.equal(copy.features?.find(f => f.id === id)?.value, value, `Feature ${id} differs`);
  }
  const original = (await client.fetchAgent({ agentId: source.id })).agent;
  assert.equal(original.persistence.sessionId, state.source_id);
  if (!args.check) {
    state.status = 'complete'; state.title = copy.title; save();
  }
  console.log(JSON.stringify({ status: args.check ? 'checked' : state.status, agent_id: copy.id,
    title: copy.title, workspace_id: copy.workspaceId, copy_id: state.copy_id, idle: true }));
} finally {
  clearTimeout(deadline);
  await client.close();
}
