import smtplib



FROM = 'kentchun33333@gmail.com'
gmail_user = FROM
gmail_pwd= '1Bxpia2a-*/'
TO = ['kentchun33333@gmail.com', 'kentchun33333@foxmail.com']
message = 'Test'
'''Not work
server = smtplib.SMTP('smtp.gmail.com:587')
server.ehlo()
server.starttls()
server.sendmail(FROM, TO, message)

server.close()
'''

'''Fail Logginb
server_ssl = smtplib.SMTP_SSL("smtp.gmail.com", 465)
server_ssl.ehlo() # optional, called by login()
server_ssl.login(gmail_user, gmail_pwd)
# ssl server doesn't support or need tls, so don't call server_ssl.starttls()
server_ssl.sendmail(FROM, TO, message)
#server_ssl.quit()
server_ssl.close()
print 'successfully sent the mail'
'''

def send_email(user, pwd, recipient, subject, body):
    import smtplib

    gmail_user = user
    gmail_pwd = pwd
    FROM = user
    TO = recipient if type(recipient) is list else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        print 'successfully sent the mail'
    except:
        print "failed to send mail"

send_email(gmail_user,gmail_pwd,TO,"TEST",message )
'''
if you want to use Port 465 you have to create an SMTP_SSL object:

# SMTP_SSL Example
server_ssl = smtplib.SMTP_SSL("smtp.gmail.com", 465)
server_ssl.ehlo() # optional, called by login()
server_ssl.login(gmail_user, gmail_pwd)
# ssl server doesn't support or need tls, so don't call server_ssl.starttls()
server_ssl.sendmail(FROM, TO, message)
#server_ssl.quit()
server_ssl.close()
print 'successfully sent the mail'
'''
