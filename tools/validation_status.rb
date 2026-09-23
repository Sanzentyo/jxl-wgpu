#!/usr/bin/env ruby
# Poll process identity and the final receipt without opening validation logs.
require 'json'
require 'open3'

def process_identity(pid)
  output, _error, status = Open3.capture3(
    { 'LC_ALL' => 'C' }, 'ps', '-p', pid.to_s, '-o', 'lstart=', '-o', 'stat='
  )
  return nil if status.exitstatus == 1 && output.strip.empty?
  raise 'process-query' unless status.success?
  match = /\A(.+\d{4})\s+(\S+)\z/.match(output.strip)
  raise 'process-query' unless match
  return nil if match[2].start_with?('Z')
  match[1].strip
end

begin
  if ARGV.length == 3 && ARGV[0] == '--record'
    directory = ARGV[1]
    raise 'existing-receipt' if File.exist?(File.join(directory, 'receipt.json'))
    pid = Integer(ARGV[2], 10)
    raise 'invalid-pid' unless pid > 1
    identity = process_identity(pid)
    raise 'process-missing' unless identity
    metadata = { 'pid' => pid, 'started' => identity }
    File.open(File.join(directory, 'process.json'), File::WRONLY | File::CREAT | File::EXCL, 0o600) do |file|
      file.write(JSON.generate(metadata))
    end
    puts 'RECORDED'
  elsif ARGV.length == 1
    directory = ARGV[0]
    metadata = JSON.parse(File.read(File.join(directory, 'process.json')))
    pid = metadata.fetch('pid')
    started = metadata.fetch('started')
    raise 'invalid-process' unless pid.is_a?(Integer) && pid > 1 && started.is_a?(String) && !started.empty?
    if process_identity(pid) == started
      puts 'RUNNING'
    else
      receipt = JSON.parse(File.read(File.join(directory, 'receipt.json')))
      code = receipt.fetch('exit')
      stable = receipt.fetch('stable')
      raise 'invalid-receipt' unless code.is_a?(Integer) && (0..255).cover?(code) && [true, false].include?(stable)
      puts "DONE exit=#{code} stable=#{stable}"
    end
  else
    warn 'usage: validation_status.rb [--record] RUN_DIRECTORY [PID]'
    exit 2
  end
rescue StandardError => error
  reason = case error
           when Errno::ENOENT then 'missing-evidence'
           when Errno::EEXIST then 'already-recorded'
           when JSON::ParserError, KeyError, TypeError, ArgumentError then 'invalid-evidence'
           when RuntimeError then error.message
           else 'unavailable'
           end
  puts "UNKNOWN #{reason}"
  exit 2
end
