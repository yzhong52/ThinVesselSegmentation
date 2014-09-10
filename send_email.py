# smtplib module send mail
 
import smtplib
 

def send_an_email():
	TO = 'yzhong.cs@gmail.com'
	SUBJECT = 'TEST MAIL'
	TEXT = 'Here is a message from python.'
	 
	# Gmail Sign In
	gmail_sender = 'vision.sharcnet@gmail.com'
	gmail_passwd = 'wsc106admin'
	 
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.ehlo()
	server.starttls()
	server.login(gmail_sender, gmail_passwd)
	 
	BODY = '\r\n'.join(['To: %s' % TO,
		            'From: %s' % gmail_sender,
		            'Subject: %s' % SUBJECT,
		            '', TEXT])
	 
	try:
	    server.sendmail(gmail_sender, [TO], BODY)
	    print ('Email sent from Python')
	except:
	    print ('Error sending mail from Python')
	 
	server.quit()
