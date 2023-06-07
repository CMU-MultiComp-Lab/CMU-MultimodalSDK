import sys
from datetime import datetime
from colorama import Fore
from tqdm import tqdm

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    OKPURPLE = '\033[0;35m'
    OKADVISORY = '\033[1;36m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def success(msgstring,destination=sys.stdout,verbose=True):
	now=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
	if type(destination) is not list:
		destination=[destination]
	if verbose==False:
		return
	for dest in destination:
		print(bcolors.OKGREEN+bcolors.BOLD+"[%s] | Success | "%now+ bcolors.ENDC+msgstring,file=dest)

def status(msgstring,destination=sys.stdout,verbose=True,end=None,require_input=False):

	now=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

	if type(destination) is not list:
		destination=[destination]

	if verbose==False:
		return

	input_from_user=None

	for dest in destination:
		if end is None:
			if require_input:
				if dest==sys.stdout:
					inp_f=raw_input if sys.version_info[0]<3 else input
					input_from_user=inp_f(bcolors.OKBLUE +bcolors.BOLD+"[%s] | Input   | "%now+bcolors.ENDC + msgstring)
				else:
					print (bcolors.OKBLUE +bcolors.BOLD+"[%s] | Status  | "%now+bcolors.ENDC + msgstring,file=dest)
					
			else:
				print (bcolors.OKBLUE +bcolors.BOLD+"[%s] | Status  | "%now+bcolors.ENDC + msgstring,file=dest)
		else:
			print (bcolors.OKBLUE +bcolors.BOLD+"[%s] | Status  | "%now+bcolors.ENDC + msgstring,file=dest,end="\r")

	if input_from_user!=None:
		return input_from_user

def advisory(msgstring,destination=sys.stdout,verbose=True):
	now=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

	if type(destination) is not list:
		destination=[destination]

	if verbose==False:
		return

	for dest in destination:
		print (bcolors.OKADVISORY +bcolors.BOLD+"[%s] | Advise  | "%now+bcolors.ENDC + msgstring,file=dest)

advise=advisory

def progress_bar(total,data=None,unit="iters",postfix="",leave=False):
	if data is None:
		return tqdm(total=total , postfix=postfix,unit=unit, leave=leave)
	#TQDM has issue with the formatting and {bar}
		#return tqdm(total=total , postfix=postfix,unit=unit, leave=leave,bar_format="%s{l_bar}%s{bar}%s{r_bar}%s" % (Fore.YELLOW,Fore.GREEN,Fore.YELLOW,Fore.RESET))
	else:
		return tqdm(data, total=total , postfix=postfix,unit=unit, leave=leave)
		#return tqdm(data, total=total , postfix=postfix,unit=unit, leave=leave,bar_format="%s{l_bar}%s{bar}%s{r_bar}%s" % (Fore.YELLOW,Fore.GREEN,Fore.YELLOW,Fore.RESET))

def error(msgstring,error=False,errorType=RuntimeError,destination=sys.stdout,verbose=True):
	now=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

	if type(destination) is not list:
		destination=[destination]

	if verbose==False:
		if error:
			raise errorType(msgstring)
		else:
			return

	if error:
		for dest in destination:
			print (bcolors.FAIL +bcolors.BOLD+"[%s] | Error   | "%now+bcolors.ENDC + msgstring,file=dest)
		raise errorType(msgstring)
	else:
		for dest in destination:
			print (bcolors.WARNING +bcolors.BOLD+"[%s] | Warning | "%now+bcolors.ENDC + msgstring,file=dest)

def warning(msgstring,destination=sys.stdout,verbose=True):
	error(msgstring=msgstring,destination=destination,verbose=verbose)

def progress_spinner(message,progress,speed=1./5000):
	speed=float(speed)
	status ("%s%s"%(message,'/-\|'[int(progress*speed)%4]),end="\r")

spinner=progress_spinner


