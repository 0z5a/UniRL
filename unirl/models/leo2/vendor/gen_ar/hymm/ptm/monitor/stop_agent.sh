ps -ef | grep teg_agent | grep -v grep|awk '{print $2}'|xargs kill -9
